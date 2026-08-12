"""Best-effort call logging.

Logging must never be able to fail a request: records go onto a bounded in-memory
queue and a background task batches them into Postgres. If the queue is full or
the database is down we drop records and count the drops — the client still gets
its answer. The drop counter is exposed on /healthz so the loss is visible rather
than silent.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

from llm_gateway.db.models import LlmAttempt, LlmCall
from llm_gateway.db.session import Database
from llm_gateway.settings import DatabaseConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AttemptRecord:
    attempt_no: int
    hop_index: int
    provider: str
    model: str
    outcome: str
    error_kind: str | None = None
    http_status: int | None = None
    latency_ms: int = 0
    backoff_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass(slots=True)
class CallRecord:
    request_id: str
    route: str
    outcome: str
    http_status: int
    api_key_id: str | None = None
    provider: str | None = None
    model: str | None = None
    error_kind: str | None = None
    attempts: int = 0
    retries: int = 0
    fallbacks: int = 0
    breaker_skips: int = 0
    hedges: int = 0
    cache_hit: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    provider_latency_ms: int = 0
    prompt_chars: int = 0
    attempt_records: list[AttemptRecord] = field(default_factory=list)


class CallRecorder:
    def __init__(self, database: Database, config: DatabaseConfig) -> None:
        self._db = database
        self._config = config
        self._queue: asyncio.Queue[CallRecord] = asyncio.Queue(maxsize=config.recorder_queue_size)
        self._task: asyncio.Task[None] | None = None
        self.dropped = 0
        self.written = 0
        self.write_failures = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="call-recorder")

    async def stop(self) -> None:
        if self._task is None:
            return
        await self.drain()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    def submit(self, record: CallRecord) -> None:
        """Never blocks and never raises: a full queue drops the record."""
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self.dropped += 1
            if self.dropped % 100 == 1:
                logger.warning("call recorder queue full, dropped %d records", self.dropped)

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait until the queue is empty — used on shutdown and by the bench harness."""
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except TimeoutError:
            logger.warning("call recorder did not drain within %.1fs", timeout)

    def stats(self) -> dict[str, Any]:
        return {
            "queued": self._queue.qsize(),
            "written": self.written,
            "dropped": self.dropped,
            "write_failures": self.write_failures,
        }

    async def _run(self) -> None:
        batch: list[CallRecord] = []
        while True:
            try:
                record = await asyncio.wait_for(
                    self._queue.get(), timeout=self._config.recorder_flush_interval_s
                )
                batch.append(record)
                while len(batch) < self._config.recorder_batch_size:
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            except TimeoutError:
                pass
            except asyncio.CancelledError:
                if batch:
                    await self._flush(batch)
                raise

            if batch:
                await self._flush(batch)
                for _ in batch:
                    self._queue.task_done()
                batch = []

    async def _flush(self, batch: list[CallRecord]) -> None:
        try:
            async with self._db.session() as session:
                session.add_all(
                    [
                        LlmCall(
                            request_id=record.request_id,
                            route=record.route,
                            api_key_id=record.api_key_id,
                            provider=record.provider,
                            model=record.model,
                            outcome=record.outcome,
                            error_kind=record.error_kind,
                            http_status=record.http_status,
                            attempts=record.attempts,
                            retries=record.retries,
                            fallbacks=record.fallbacks,
                            breaker_skips=record.breaker_skips,
                            hedges=record.hedges,
                            cache_hit=record.cache_hit,
                            tokens_in=record.tokens_in,
                            tokens_out=record.tokens_out,
                            cost_usd=record.cost_usd,
                            latency_ms=record.latency_ms,
                            provider_latency_ms=record.provider_latency_ms,
                            prompt_chars=record.prompt_chars,
                        )
                        for record in batch
                    ]
                )
                session.add_all(
                    [
                        LlmAttempt(
                            request_id=record.request_id,
                            attempt_no=attempt.attempt_no,
                            hop_index=attempt.hop_index,
                            provider=attempt.provider,
                            model=attempt.model,
                            outcome=attempt.outcome,
                            error_kind=attempt.error_kind,
                            http_status=attempt.http_status,
                            latency_ms=attempt.latency_ms,
                            backoff_ms=attempt.backoff_ms,
                            tokens_in=attempt.tokens_in,
                            tokens_out=attempt.tokens_out,
                            cost_usd=attempt.cost_usd,
                        )
                        for record in batch
                        for attempt in record.attempt_records
                    ]
                )
                await session.commit()
            self.written += len(batch)
        except Exception:
            self.write_failures += 1
            logger.exception("failed to persist %d call records", len(batch))
