"""Synthetic workload for chaos runs.

Real traffic repeats itself: a handful of questions account for most requests, and
the same question arrives in slightly different wording. The generator reproduces
that shape — a Zipf-weighted topic distribution with four phrasings per topic — so
the cache hit rate measured in RELIABILITY.md means something. A uniform stream of
unique prompts would report a 0% hit rate and prove nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

#: Each entry is one topic in four phrasings. Requests that share a topic should be
#: cache-equivalent; requests from different topics should not be.
TOPICS: list[list[str]] = [
    [
        "What is the capital of France?",
        "whats the capital of france",
        "Tell me the capital city of France.",
        "France — what's its capital?",
    ],
    [
        "How do I reverse a string in Python?",
        "python reverse a string how",
        "What's the way to reverse a string in Python?",
        "Show me how to reverse a Python string.",
    ],
    [
        "Explain what a database index is.",
        "what is a database index",
        "Can you explain database indexes?",
        "Describe what an index does in a database.",
    ],
    [
        "What does HTTP status 429 mean?",
        "http 429 meaning",
        "Explain the 429 status code.",
        "What is a 429 response?",
    ],
    [
        "Summarise the difference between TCP and UDP.",
        "tcp vs udp difference",
        "How do TCP and UDP differ?",
        "Compare TCP and UDP for me.",
    ],
    [
        "How does exponential backoff work?",
        "explain exponential backoff",
        "What is exponential backoff with jitter?",
        "Describe the exponential backoff retry strategy.",
    ],
    [
        "What is a circuit breaker in distributed systems?",
        "circuit breaker pattern explanation",
        "Explain the circuit breaker pattern.",
        "How does a circuit breaker protect a service?",
    ],
    [
        "Write a haiku about autumn rain.",
        "haiku about rain in autumn",
        "Compose a short haiku on autumn rain.",
        "Autumn rain haiku please.",
    ],
    [
        "What is the time complexity of quicksort?",
        "quicksort time complexity",
        "How fast is quicksort?",
        "Explain quicksort's complexity.",
    ],
    [
        "How do I set up a Postgres connection pool?",
        "postgres connection pool setup",
        "What's the right way to pool Postgres connections?",
        "Explain Postgres connection pooling.",
    ],
    [
        "What is idempotency in an API?",
        "api idempotency meaning",
        "Explain idempotent API requests.",
        "Why should an API be idempotent?",
    ],
    [
        "Describe the difference between latency and throughput.",
        "latency vs throughput",
        "How are latency and throughput different?",
        "Explain throughput compared to latency.",
    ],
    [
        "What is a vector embedding?",
        "explain vector embeddings",
        "How do embeddings represent text?",
        "Describe what an embedding vector is.",
    ],
    [
        "How does Docker layer caching work?",
        "docker layer cache explanation",
        "Explain caching of Docker layers.",
        "Why do Docker builds reuse layers?",
    ],
    [
        "What is the difference between a mutex and a semaphore?",
        "mutex vs semaphore",
        "Compare mutexes and semaphores.",
        "How does a semaphore differ from a mutex?",
    ],
    [
        "Explain what a p95 latency number means.",
        "what does p95 latency mean",
        "How should I read a p95 latency metric?",
        "Describe the 95th percentile of latency.",
    ],
    [
        "How do I write a good commit message?",
        "good git commit message tips",
        "What makes a commit message useful?",
        "Explain how to write commit messages well.",
    ],
    [
        "What is CORS and why does it block my request?",
        "cors explanation why blocked",
        "Explain CORS errors in the browser.",
        "Why am I getting a CORS error?",
    ],
    [
        "Describe blue-green deployment.",
        "blue green deployment explained",
        "What is a blue-green release?",
        "How does blue-green deployment work?",
    ],
    [
        "What is the CAP theorem?",
        "cap theorem explanation",
        "Explain CAP in distributed databases.",
        "Describe consistency, availability and partition tolerance.",
    ],
    [
        "How do I profile a slow Python function?",
        "profile slow python code",
        "What tools profile Python performance?",
        "Explain how to find a slow Python function.",
    ],
    [
        "What is a dead letter queue?",
        "dead letter queue meaning",
        "Explain dead letter queues in messaging.",
        "Why use a dead letter queue?",
    ],
    [
        "Explain the difference between authentication and authorization.",
        "authn vs authz difference",
        "How is authorization different from authentication?",
        "Compare authentication and authorization.",
    ],
    [
        "What does eventual consistency mean?",
        "eventual consistency explained",
        "Explain eventually consistent systems.",
        "How does eventual consistency behave?",
    ],
]


@dataclass(slots=True)
class WorkloadStats:
    requests: int
    unique_prompts: int
    unique_topics: int
    duplicate_rate: float
    paraphrase_rate: float


def build_workload(
    count: int,
    *,
    seed: int = 42,
    zipf_exponent: float = 1.1,
) -> tuple[list[str], WorkloadStats]:
    """Return `count` prompts drawn from a Zipf-weighted topic distribution."""
    rng = random.Random(seed)
    weights = [1.0 / ((index + 1) ** zipf_exponent) for index in range(len(TOPICS))]

    prompts: list[str] = []
    topics_used: list[int] = []
    for _ in range(count):
        topic_index = rng.choices(range(len(TOPICS)), weights=weights, k=1)[0]
        variants = TOPICS[topic_index]
        prompts.append(variants[rng.randrange(len(variants))])
        topics_used.append(topic_index)

    unique_prompts = len(set(prompts))
    unique_topics = len(set(topics_used))
    return prompts, WorkloadStats(
        requests=count,
        unique_prompts=unique_prompts,
        unique_topics=unique_topics,
        duplicate_rate=round(1 - unique_prompts / count, 4) if count else 0.0,
        # Share of requests whose topic was already seen: the ceiling for a
        # semantic cache that matches paraphrases rather than exact strings.
        paraphrase_rate=round(1 - unique_topics / count, 4) if count else 0.0,
    )
