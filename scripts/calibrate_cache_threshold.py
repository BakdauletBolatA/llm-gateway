#!/usr/bin/env python
"""Pick the semantic-cache similarity threshold from measurement, not from taste.

Prints two things:

1. The separation of the embedder on the chaos workload: how similar two
   phrasings of the *same* question are, versus two *different* questions.
   A usable threshold has to sit above the cross-topic maximum.
2. A replay of the workload through a simulated cache at several thresholds,
   with the number of answers that would have been served from a different
   topic — i.e. wrong answers.

    python scripts/calibrate_cache_threshold.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chaos.workload import TOPICS, build_workload  # noqa: E402
from llm_gateway.cache.embedder import HashingEmbedder, cosine_similarity  # noqa: E402

THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 0.93]


def main() -> int:
    embedder = HashingEmbedder(256)
    vectors = [[embedder.embed_sync(variant) for variant in topic] for topic in TOPICS]

    within = sorted(
        cosine_similarity(a, b) for topic in vectors for a, b in combinations(topic, 2)
    )
    cross = sorted(
        cosine_similarity(a, b)
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
        for a in vectors[i]
        for b in vectors[j]
    )

    print(f"embedder: {embedder.name}, dim={embedder.dim}")
    print(
        f"same question, different phrasing (n={len(within)}): "
        f"min={within[0]:.3f} median={within[len(within) // 2]:.3f} max={within[-1]:.3f}"
    )
    print(
        f"different questions       (n={len(cross)}): "
        f"median={cross[len(cross) // 2]:.3f} p99={cross[int(len(cross) * 0.99)]:.3f} "
        f"max={cross[-1]:.3f}"
    )
    print(f"\nany threshold must exceed {cross[-1]:.3f} to avoid answering the wrong question\n")

    prompts, stats = build_workload(150, seed=42)
    topic_of = {variant: index for index, topic in enumerate(TOPICS) for variant in topic}
    print(
        f"workload replay: {stats.requests} requests, {stats.unique_prompts} unique prompts, "
        f"{stats.unique_topics} unique topics"
    )
    print(f"{'threshold':>10} {'hit rate':>9} {'entries':>8} {'wrong answers':>14}")
    for threshold in THRESHOLDS:
        store: list[tuple[list[float], str]] = []
        hits = 0
        wrong = 0
        for prompt in prompts:
            vector = embedder.embed_sync(prompt)
            best_prompt, best_score = None, -2.0
            for stored_vector, stored_prompt in store:
                score = cosine_similarity(vector, stored_vector)
                if score > best_score:
                    best_score, best_prompt = score, stored_prompt
            if best_prompt is not None and best_score >= threshold:
                hits += 1
                if topic_of[best_prompt] != topic_of[prompt]:
                    wrong += 1
            else:
                store.append((vector, prompt))
        print(
            f"{threshold:>10.2f} {hits / len(prompts):>8.1%} {len(store):>8} {wrong:>14}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
