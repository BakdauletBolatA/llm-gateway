# llm-gateway

An OpenAI-compatible gateway in front of Anthropic, OpenAI and Ollama, with spend
tracking in PostgreSQL, a hard budget, and the reliability mechanisms you need
when a provider is slow, full, or down: timeouts, retries, circuit breaker,
fallback, hedging, concurrency limits, rate limiting, a response cache, and a
router that sends easy requests to a small model.

**The point of the project is not that it works, but that every claim about it
comes with a measurement you can re-run.** The gateway was built in iterations:
a deliberately naive version first, then one reliability mechanism at a time, each
measured on the same load. The results below are generated from files in
[`reports/`](reports/) and [`bench/results/`](bench/results/) by scripts in this
repository, and a test fails if the README drifts from them.

> The long-form report, [RELIABILITY.md](RELIABILITY.md), is written in Russian. Its
> tables are generated from `bench/results` and checked by tests.

## The problem

An application that calls one LLM provider inherits that provider's bad days:
latency tails, rate limits, outages, and a bill nobody capped. Putting a gateway
in front fixes that only if the gateway's own mechanisms are right, and several of
them make things worse when they are wrong: a circuit breaker without a fallback
lowers availability; a cache that answers a *similar* question returns the wrong
answer; a hedge doubles load on a provider that is already full. This repository
builds those mechanisms and measures what each one is worth.

## Architecture

```mermaid
flowchart LR
    client([Client]) -->|POST /v1/chat/completions| auth[Auth and rate limit]
    auth --> auto{model = auto?}
    auto -->|yes| complexity[Complexity router<br/>small or large route]
    auto -->|no| cache
    complexity --> cache{Cache<br/>opt-in, per tenant}
    cache -->|hit| client
    cache -->|miss| budget[Budget reservation]
    budget -->|limit reached| refuse([402, no provider call])
    budget --> chain

    subgraph chain [Provider chain]
        direction LR
        breaker[Circuit breaker] --> bulkhead[Concurrency limit] --> retry[Retries with jitter]
        retry -.->|hedge after a delay| next[Next provider]
    end

    chain --> openai[OpenAI dialect]
    chain --> anthropic[Anthropic dialect]
    chain --> ollama[Ollama dialect]

    gateway[(PostgreSQL + pgvector<br/>calls, spend, cache)]
    chain -.-> gateway
    cache -.-> gateway
    budget -.-> gateway
    chain -->|/metrics| prom[Prometheus] --> grafana[Grafana]
```

Everything the gateway does to a request is in one file,
[`src/llm_gateway/router.py`](src/llm_gateway/router.py). Each mechanism is switched
by a flag in the config, so the naive baseline and the final build are the same code
with different settings. That is what lets any row of the report be re-run without
checking out an old commit.

| mechanism | what it does | config |
|---|---|---|
| timeouts | connect/read/write per call plus a deadline for the whole request | `reliability.timeouts` |
| retries | exponential backoff with jitter, honours `Retry-After` | `reliability.retries` |
| circuit breaker | per provider, sliding window, half-open probes | `reliability.circuit_breaker` |
| fallback | an ordered chain of providers per route, across API dialects | `reliability.fallback` |
| hedging | duplicate to the next provider when the current one is silent too long | `reliability.hedging` |
| concurrency limit | at most N calls in flight per provider, queue bounded by the deadline | `reliability.bulkhead` |
| rate limit | token bucket per API key, `429` before any work; shared across replicas through Postgres | `reliability.rate_limit` |
| response cache | only when the client sends `"cache": true`; scoped to route, model, API key, generation parameters and conversation context; exact match by default, embedding match optional | `reliability.cache` |
| budget | `402` before the provider is called; the amount is reserved up front, in Postgres when shared across replicas | `budget` |
| complexity routing | `model: "auto"` goes to a small or a large route by transparent rules; the decision and its reasons come back with the response | `routing.complexity` |

## Run it

From a clean checkout, with Docker only. No API key is needed: the default route
talks to the bundled mock provider.

```bash
docker compose up -d --build

curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"chaos-default","messages":[{"role":"user","content":"Hello"}]}' | jq

curl -s 'localhost:8080/v1/usage?group_by=provider' | jq
```

**With a real local model** (the images and weights are a few gigabytes, so it is a
profile). `qwen2.5:0.5b` is pulled automatically and runs on a CPU:

```bash
OLLAMA_ENABLED=true GATEWAY_CONFIG_OVERLAY=config/extras/live_local.yaml \
  docker compose --profile ollama up -d --build

curl -si localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is the capital of Australia?"}]}' \
  | grep -i '^x-gateway-\(route\|model\)'
```

**With Prometheus and Grafana** (dashboard provisioned, no login for viewing,
bound to localhost):

```bash
docker compose --profile observability up -d     # Grafana on http://127.0.0.1:3000
```

**Two replicas on one Postgres**, to see the difference between a per-process and a
shared limit:

```bash
RATE_LIMIT_SCOPE=shared BUDGET_SCOPE=shared docker compose --profile replica up -d --build
python scripts/shared_limit_probe.py \
  --gateway http://127.0.0.1:8080 --gateway http://127.0.0.1:8082
```

**Paid providers**: copy `.env.example` to `.env`, set the keys and
`OPENAI_ENABLED=true` / `ANTHROPIC_ENABLED=true`, and use the `production` route.

**Without Docker** (needs an external PostgreSQL with `pgvector`):

```bash
make install
DATABASE_URL=postgresql+asyncpg://gateway:gateway@127.0.0.1:5432/llm_gateway \
MOCK_BASE_URL=http://127.0.0.1:8081 make native-up
```

## Results

Every number below is produced by a script in this repository. To reproduce them:

```bash
pip install -e ".[dev,embeddings,loadtest]"
python eval.py --start-stack          # cache + routing (replayed) + load test on the mock
python eval.py --live                 # additionally re-measure on live local models (slow)
python eval.py --check                # verify this table matches reports/
```

`eval.py` runs [`eval/cache_eval.py`](eval/cache_eval.py),
[`eval/routing_eval.py`](eval/routing_eval.py) and
[`loadtest/run.py`](loadtest/run.py), writes machine-readable files to
[`reports/`](reports/), and rebuilds the block below with
[`eval/readme_table.py`](eval/readme_table.py). The chaos benchmark has its own
script, [`scripts/reproduce_report.sh`](scripts/reproduce_report.sh) (about 20
minutes).

<!-- results:start -->

**Reliability under injected failures** — mock provider, 11 failure profiles ([`chaos/run.py`](src/chaos/run.py), full report in [RELIABILITY.md](RELIABILITY.md)):

| build (mock provider) | success, all scenarios | scenarios at 100% | p95, `storm` | p95, `hang` | cost, all requests |
|---|---|---|---|---|---|
| naive gateway | 46.4% | 2 of 11 | 20,002 ms | 20,003 ms | $0.0494 |
| final build | 90.9% | 10 of 11 | 250 ms | 520 ms | $0.2123 |

**Semantic cache** — 121 labelled query pairs, hand-written ([`eval/cache_eval.py`](eval/cache_eval.py)). The threshold is chosen on half the pairs and scored on the other half:

| embedder | highest similarity of a *different* pair | threshold at 5% false hits | held-out hit rate | held-out false hits |
|---|---|---|---|---|
| hash n-gram vectors (the old matcher) | 0.857 | 0.86 | 0.0% | 0 of 30 |
| all-MiniLM-L6-v2 (local, CPU) | 0.995 | 0.93 | 12.9% | 1 of 30 |

**Load test** — closed loop, `max_tokens` 64 ([`loadtest/run.py`](loadtest/run.py)). *kill* stops the primary backend halfway. With several runs a cell is the median, with the minimum and maximum in brackets:

| backend | scenario | users | runs | req/s | p50 | p95 | p99 | success | failover |
|---|---|---|---|---|---|---|---|---|---|
| mock provider | steady | 4 | 5 | 43.85 (43.45-44.54) | 90 ms (89 ms-91 ms) | 127 ms (126 ms-128 ms) | 134 ms (131 ms-135 ms) | 100.0% (100.0%-100.0%) |  |
| mock provider | kill | 4 | 5 | 42.33 (42.05-42.52) | 89 ms (89 ms-90 ms) | 130 ms (130 ms-134 ms) | 233 ms (230 ms-240 ms) | 100.0% (100.0%-100.0%) | failed after the kill: 0 (0-0); first answer from the other server: 0.0 (0.0-0.1) s |
| live qwen2.5:0.5b, CPU | steady | 4 | 5 | 0.64 (0.52-0.68) | 6,110 ms (5,937 ms-6,629 ms) | 7,537 ms (6,955 ms-15,536 ms) | 7,938 ms (7,410 ms-18,025 ms) | 100.0% (100.0%-100.0%) |  |
| live qwen2.5:0.5b, CPU | kill | 4 | 5 | 0.64 (0.61-0.78) | 5,882 ms (4,620 ms-6,609 ms) | 7,937 ms (7,124 ms-9,092 ms) | 10,802 ms (8,676 ms-11,798 ms) | 100.0% (100.0%-100.0%) | failed after the kill: 0 (0-0); first answer from the other server: 1.7 (0.4-2.2) s |

**Routing by complexity** — correctness is a programmatic check, cost is modeled from measured tokens at gpt-4o-mini / gpt-4o prices ([`eval/routing_eval.py`](eval/routing_eval.py)):

| set (live qwen2.5:0.5b and 3b) | correct, always large | correct, always small | correct, routed | modeled $/100, always large | modeled $/100, routed | sent to small | router vs hand labels |
|---|---|---|---|---|---|---|---|
| 50 prompts, rules as first run | 49/50 | 41/50 | 43/50 | $0.0336 | $0.0188 | 62.0% | 88.0% |
| 50 prompts, rules tuned on this set | 49/50 | 41/50 | 47/50 | $0.0336 | $0.0249 | 52.0% | 98.0% |
| 30 held-out prompts, rules frozen | 29/30 | 25/30 | 28/30 | $0.0319 | $0.0215 | 63.3% | 86.7% |

<!-- results:end -->

### What was measured on the mock, and what on a live model

| measurement | backend | what that means |
|---|---|---|
| chaos benchmark: 11 failure profiles, 9 iterations, ablations, multi-replica probes | **mock provider** | failure behaviour is injected and deterministic, so rows are comparable between iterations. A live model gives hardware-dependent latency, which would make them incomparable. |
| load test, mock rows | **mock provider** | the same two scenarios and the same reliability overlay as the live rows, so the difference between the two is the backend |
| load test, live rows | **live `qwen2.5:0.5b`** on Ollama, CPU only, Docker on an 8-core laptop | 5 runs per row of 90 s (about 55-70 requests each), reported as median with the range. The p99 of a run is close to its maximum. |
| routing eval | **live `qwen2.5:0.5b` and `qwen2.5:3b`** | answers were recorded once, then every policy is replayed over the same answers |
| cache eval | **local `all-MiniLM-L6-v2`** (and the hash n-gram embedder for comparison) | no provider involved |
| adapters for Anthropic and OpenAI | **neither**; unit-tested against `httpx.MockTransport` | request format, response parsing and error classification are tested; the chaos harness has never been run against a paid API |

Limits that apply to the tables above:

- The load test uses 4 virtual users: a CPU-bound 0.5B model is already saturated
  there. There are no runs at higher concurrency.
- Live throughput varies more between sessions than within one. The first single
  run of the *steady* scenario measured 1.04 req/s
  ([`reports/loadtest_live_steady.json`](reports/loadtest_live_steady.json)); five
  repeats taken later on the same laptop gave 0.52-0.68, so that first run was
  outside the range of the repeats. A *kill* run made while other applications were
  using the CPU managed 0.24 req/s instead of 0.98
  ([`reports/loadtest_live_kill_contended.json`](reports/loadtest_live_kill_contended.json)).
  Read the live rows as "about 0.5-1 req/s on this hardware", not as a benchmark of
  the gateway: the gateway adds almost nothing, the model is the bottleneck. The cause
  of the gap between sessions was not identified. No failed requests in any run.
- The live model is warmed on both Ollama servers before timing starts, so a cold
  start on the failover target is not in the numbers.
- Routing cost is **modeled**, not billed: tokens are measured on the local models
  and priced at gpt-4o-mini and gpt-4o list prices from `config/gateway.yaml`.
  Running locally costs nothing, and the two tokenizers differ, so read it as an
  order of magnitude.
- Routing correctness is a programmatic check (the answer contains the right fact,
  number or code construct). It is not a quality score, and generated code is
  never executed. The simple/complex labels are the author's judgement; edit
  `eval/data/routing_prompts.jsonl` to apply yours.
- The cache and routing datasets are small and written by hand, so confidence
  intervals are wide. No LLM judge is used, so there is no judge-versus-human
  agreement to report.

## What failed, and what I learned

**The cache served answers to different questions, and to different people.** The
first cache matched on hash n-gram vectors at a similarity threshold of 0.60. That
threshold was calibrated on the benchmark workload (22 deliberately different
topics), where nothing collides. On ordinary questions it does:
`What is the capital of France?` / `What is the capital of Spain?` score 0.775 and
`Is it safe to take ibuprofen with alcohol?` / `...without alcohol?` score 0.852, both
above the threshold and above some real paraphrases. The cache was also on by
default and keyed on route and model only, so one tenant's answer could be served to
another, as could an answer written for a different system prompt. All of it is
fixed (exact matching, opt-in, scoped per tenant, context and generation
parameters), with a failing test for each, and the benchmark scripts now switch the
old matcher on explicitly.

**A real embedding model does not make a similarity threshold safe.** With
`all-MiniLM-L6-v2`, `How do I convert Celsius to Fahrenheit?` and the reverse score
0.995, and `enable dark mode` / `disable dark mode` score 0.927, higher than most
genuine paraphrases. On the 121-pair set no threshold has zero false hits; at an
accepted 5% false-hit rate the held-out hit rate is 12.9%. The default is therefore
exact matching, the cache is off in the shipped config, and the semantic mode is
documented as something you enable knowingly for low-stakes traffic.

**The first version of the router lost answers.** On the first run, routing scored
43 of 50 against 49 of 50 for always using the large model, and the 6 prompts it
sent to the small model by mistake were arithmetic word problems. The cause was a
real bug: the pattern `\d+\s*%\b` can never match, because `%` is not a word
character. I fixed it, and then the rules scored 47 of 50 on the same prompts, which
is optimistic because I tuned on the failures. That is why the table also has 30
prompts written and committed before the router was run on them, with the rules
frozen: 28 of 30 against 29 of 30 for the large model, at a third lower modeled
cost. The weakness that remains is a short hard question with no keyword, such as
`Is 91 a prime number?`: rules over surface features cannot see it.

**`docker compose up` did not work from a clean machine.** The image installs the
package into `site-packages`, but migrations were located relative to the source
file, so the gateway crashed at startup in the container. Tests ran from the
checkout and never noticed. Found while preparing the load test, fixed with a test.

**Two of my own grading criteria were wrong.** After the first routing run, two
checks rejected correct answers from the large model (a phrase list that was too
narrow, and a `max()` mentioned in prose rather than in code). I corrected them and
said so in the report: the corrected criteria move the large model from 47 to 49 of
50.

**Mechanisms interact.** The most useful results of the benchmark were the
counterintuitive ones: a circuit breaker without a fallback lowers availability;
a provider that is merely full looks broken to a breaker unless `Retry-After` is
treated as backpressure; a hedge and a concurrency limit cancel each other on a
provider with a quota; and a short benchmark measures warm-up, not steady state.
Each is measured in [RELIABILITY.md](RELIABILITY.md).

## What is deliberately not done

- **No streaming.** Retries and fallback after the first token arrive are a separate
  problem with a different error model.
- **No "not a cent over".** The budget reserves an *estimate* up front and settles
  to the actual cost, so what remains is requests in flight times the estimation
  error. Measured: +4.9% on one replica, +5.6% on two with `BUDGET_SCOPE=shared`
  (and +107.9% on two replicas with the per-process counter). Closing it needs a
  conservative estimate, at the price of leaving budget unused (measured −44.2%).
- **No safe semantic cache by default.** See above.

## Configuring it for your provider

The defaults are tuned for the bundled mock. The measurements give rules for
re-deriving them:

| if your provider… | set | evidence |
|---|---|---|
| has a normal p95 of X | `hedging.delay_ms` ≈ X | below X you duplicate healthy traffic and p95 rises |
| has a concurrency quota N | `providers.<name>.max_concurrent: N` and **no hedging** on that route | a cancelled duplicate frees a slot on your side, not on the provider's |
| is ordered cheap to expensive | remember that hedging shifts traffic to the next hop | on the `slow` profile it costs ×8.6 per answer |
| sends `Retry-After` | `circuit_breaker.retry_after_is_backpressure: true` | otherwise a busy provider looks dead |
| has a known capacity | `rate_limit.requests_per_second` from it | a refusal in 16 ms beats a 12 s timeout |
| runs on more than one replica | `RATE_LIMIT_SCOPE=shared` and `BUDGET_SCOPE=shared` | otherwise each replica grants a full limit |
| returns long answers | `budget.estimate_output_tokens` with headroom | an estimate is reserved and the actual is paid: too low overspends, too high under-uses the limit |
| has a short `cache.ttl_s` and dense traffic | `cache.sweep_interval_s` no larger than the TTL | expired rows are not removed by themselves; 20,000 of them slow a lookup from 3.6 ms to 13.5 ms |
| runs on a CPU-bound local model | `config/extras/live_local.yaml` | long timeouts, no hedging, a small concurrency limit |

## See it work

Start the stack with a live model and the dashboard (`--build` matters: a stale image
answers with the old routing rules), then open <http://127.0.0.1:3000>:

```bash
OLLAMA_ENABLED=true GATEWAY_CONFIG_OVERLAY=config/extras/live_local.yaml \
  docker compose --profile ollama --profile observability up -d --build
```

**Failover.** Run load and let the runner stop the first Ollama server halfway
(`docker compose --profile ollama start ollama` brings it back):

```bash
python loadtest/run.py --target live --scenario kill --duration 40
```

In Grafana, *Circuit breaker state* for `ollama` goes to 2, *Provider calls by outcome*
moves from `ollama` to `ollama_secondary`, and the success rate stays at 100%.

**Routing.** The decision and its reasons come back in the headers:

```bash
curl -si localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is 15% of 240? End with Answer: <number>"}]}' \
  | grep -i '^x-gateway-\(route\|model\)'
```

A hard question goes to the 3B model; `What is the capital of Australia?` goes to the
0.5B one with `score=0`. Grafana panel titles are in Russian, the metric names are not.
Run this on an otherwise idle machine: the live model is CPU-bound.

## API

| endpoint | purpose | key |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible request. `model` is a **route** from the config, or `auto`; `"cache": true` opts in to cached answers | yes |
| `GET /v1/usage` | spend for the period: `?from=&to=&group_by=provider\|model\|route\|api_key\|day` | yes |
| `GET /v1/reliability/state` | breaker states, cache statistics, budget | yes |
| `POST /v1/reliability/reset` | reset breakers (and the cache with `?cache=true`) between runs | yes |
| `GET /v1/config` | effective reliability config, no secrets | no |
| `GET /metrics` | Prometheus metrics | no |
| `GET /healthz`, `GET /readyz` | liveness and readiness (checks the database) | no |

The *key* column applies only when `auth.keys` is non-empty; by default
authentication is off. The line is drawn by what an endpoint exposes rather than by
whether it is operational: `/v1/reliability/state` reveals per-key spend, and
`reset?cache=true` empties the cache so every following request reaches a paid
provider. `/metrics` stays behind the network policy, not a key.

Every response carries its own telemetry in headers, which is how the chaos harness
and the load test measure without reading the database:

```
X-Gateway-Route: large-local                    X-Gateway-Provider: ollama
X-Gateway-Route-Decision: large; score=3; reasons=math(15%):+3
X-Gateway-Attempts: 1      X-Gateway-Retries: 0       X-Gateway-Fallbacks: 0
X-Gateway-Breaker-Skips: 0 X-Gateway-Hedges: 0        X-Gateway-Cache: miss
X-Gateway-Cost-Usd: 0.000000                    X-Gateway-Latency-Ms: 3874
```

Errors are typed: `429` (provider limit or our own rate limit), `402` (budget
exhausted), `503` (breaker open, no free slot, or no enabled provider), `504`
(timeout), `502` (broken provider), `400` (bad request). The body always has
`error.kind` from one taxonomy, the `request_id` and the number of attempts.

### Observability

The same fact about a request goes to three places and must not disagree: response
headers, the Postgres log of calls and attempts, and `/metrics`. The metrics are built
from the same attempt records as the database rows, not from separate counters.

```bash
curl -s localhost:8080/metrics | grep -E '^llm_gateway_(requests|provider_calls|cost|routing)'
```

| metric | what it is for |
|---|---|
| `llm_gateway_requests_total{route,outcome}` | availability per route |
| `llm_gateway_request_errors_total{route,kind}` | why requests fail |
| `llm_gateway_request_duration_seconds{route}` | histogram with buckets chosen for this service |
| `llm_gateway_provider_calls_total{provider,outcome}` | `success`, `error`, `cancelled` (lost a hedge), `discarded`, `skipped_breaker` |
| `llm_gateway_routing_decisions_total{tier}` | how many `auto` requests went to the small and the large route |
| `llm_gateway_retries_total`, `_fallbacks_total`, `_hedges_total` | work created by the reliability mechanisms |
| `llm_gateway_cost_usd_total{provider}` | money by provider |
| `llm_gateway_circuit_breaker_state{provider}` | 0 closed, 1 half-open, 2 open |
| `llm_gateway_budget_spent_usd` / `_limit_usd` | how close the budget is to refusing |
| `llm_gateway_budget_shared` / `_backend_errors` | whether the limit is shared across replicas and whether reservations are failing |
| `llm_gateway_cache_expired_swept` | stuck at zero with a live cache means the sweep is broken |
| `llm_gateway_recorder_queue_depth` / `_records_dropped` | whether the call log is losing records |

The dashboard in [`ops/grafana-dashboard.json`](ops/grafana-dashboard.json) is
provisioned by the `observability` profile and can also be imported into any Grafana.
`LOG_FORMAT=json` switches logs, including the uvicorn access log, to one JSON
object per line.

## Chaos testing

```bash
scripts/reproduce_report.sh                        # the whole report from scratch, ~20 min
ONLY=09 scripts/reproduce_report.sh                # re-run one iteration
python -m chaos.run --label my_run --all           # all 11 scenarios
python -m chaos.run --label my_run --scenario storm --n 300 --concurrency 20
python -m chaos.report                             # rebuild the tables in RELIABILITY.md
scripts/run_ablations.sh                           # switch mechanisms off one at a time
```

`reproduce_report.sh` overwrites `bench/results`; that is its purpose. To test the
script without touching the measurements: `OUT=/tmp/probe N=10 SETTLE=0
scripts/reproduce_report.sh`.

Failure profiles and scenarios are defined in `config/failure_profiles.yaml`; the mock
knows *how* to fail, not *when*. Injection is deterministic: the deck of outcomes is
seeded from `(seed, upstream, profile)`, so a re-run gives the same proportions.

## Development

```bash
make install     # venv and dependencies (uv)
make test        # pytest: unit plus end-to-end through ASGI against an in-process mock
make lint        # ruff check, ruff format --check, mypy
make bench LABEL=my_run
```

The tests need PostgreSQL with `pgvector`. Tests that need it **skip themselves when
the database is missing** (`TEST_DATABASE_URL`, default `llm_gateway_test`), so a
green run without Postgres is not the whole suite: CI runs it with a Postgres
service. End-to-end tests connect the gateway to the mock through an ASGI transport,
so the whole request path is real, without a network and without keys.

```
src/llm_gateway/     the gateway: routing, provider adapters, reliability, database, budget
src/mock_provider/   the mock: three API dialects, failure profiles
src/chaos/           chaos harness and the report table generator
eval/                cache and routing evals, datasets, README table generator
loadtest/            Locust scenario and the runner that writes reports/
eval.py              runs every measurement and rebuilds the README table
config/              gateway.yaml, failure profiles, iteration, ablation and extra overlays
bench/results/       chaos benchmark results (JSON); RELIABILITY.md is built from them
reports/             eval and load-test results (JSON)
migrations/          Alembic, applied automatically on startup
scripts/             native stack, report reproduction, ablations, CI smoke run
ops/                 Prometheus config, Grafana provisioning and dashboard
```
