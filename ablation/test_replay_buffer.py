#!/usr/bin/env python
# coding=utf-8
"""
Test the experience replay buffer (CPU logic + optional GPU end-to-end).

Part 1 (CPU only, no GPU needed):
  - Creates a ReplayBuffer, adds synthetic entries, samples, verifies
    persistence (save / load), tests FIFO eviction, and tests
    mix_replay_into_batch.

Part 2 (optional, requires 1 GPU):
  - Runs a small difficulty estimation, identifies Hard-bucket prompts,
    generates rollouts, adds breakthrough trajectories to the buffer,
    then demonstrates batch mixing.

Usage:
    # CPU-only unit tests:
    python -m ablation.test_replay_buffer

    # With GPU end-to-end (pass --model_path):
    CUDA_VISIBLE_DEVICES=0 python -m ablation.test_replay_buffer \
        --model_path /path/to/Qwen2.5-Math-1.5B \
        --dataset_key math_500 \
        --num_problems 10 \
        --num_generations 4 \
        --output_dir outputs/ablation/replay_buffer
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_replay_buffer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test replay buffer (CPU unit tests + optional GPU end-to-end)"
    )
    parser.add_argument("--model_path", type=str, default=None,
                        help="If provided, runs the GPU end-to-end test.")
    parser.add_argument("--dataset_key", type=str, default="math_500")
    parser.add_argument("--num_problems", type=int, default=10)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation/replay_buffer")
    return parser.parse_args()


# ======================================================================
# Part 1: CPU unit tests
# ======================================================================

def test_basic_add_and_sample() -> None:
    """Test adding entries and sampling from the buffer."""
    from utils.replay_buffer import ReplayBuffer

    with tempfile.TemporaryDirectory() as tmpdir:
        buf = ReplayBuffer(os.path.join(tmpdir, "buf.json"), max_size=100)

        assert len(buf) == 0, f"Expected empty buffer, got {len(buf)}"

        for i in range(20):
            buf.add(
                prompt=f"prompt_{i}",
                completion=f"completion_{i}",
                prompt_id=f"id_{i}",
                difficulty_score=0.85 + i * 0.005,
                reward=1.0,
            )

        assert len(buf) == 20, f"Expected 20 entries, got {len(buf)}"

        samples = buf.sample(5, rng=random.Random(42))
        assert len(samples) == 5, f"Expected 5 samples, got {len(samples)}"
        for s in samples:
            assert s.prompt.startswith("prompt_"), f"Unexpected prompt: {s.prompt}"

        logger.info("  PASS: basic add and sample")


def test_fifo_eviction() -> None:
    """Test that old entries are evicted when max_size is exceeded."""
    from utils.replay_buffer import ReplayBuffer

    with tempfile.TemporaryDirectory() as tmpdir:
        buf = ReplayBuffer(os.path.join(tmpdir, "buf.json"), max_size=5)

        for i in range(10):
            buf.add(prompt=f"p{i}", completion=f"c{i}", difficulty_score=0.9)

        assert len(buf) == 5, f"Expected 5 entries after eviction, got {len(buf)}"

        all_entries = buf.sample(5)
        prompts = {e.prompt for e in all_entries}
        assert "p0" not in prompts, "Old entry p0 should have been evicted"
        assert "p9" in prompts, "Recent entry p9 should be present"

        logger.info("  PASS: FIFO eviction")


def test_persistence() -> None:
    """Test save/load roundtrip."""
    from utils.replay_buffer import ReplayBuffer

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "buf.json")
        buf = ReplayBuffer(path, max_size=50, hard_threshold=0.75)

        for i in range(8):
            buf.add(
                prompt=f"prompt_{i}",
                completion=f"completion_{i}",
                prompt_id=f"id_{i}",
                difficulty_score=0.8 + i * 0.01,
                reward=1.0,
                metadata={"step": i},
            )
        buf.save()

        buf2 = ReplayBuffer(path)
        assert len(buf2) == 8, f"Expected 8 entries after reload, got {len(buf2)}"
        assert buf2.max_size == 50
        assert buf2.hard_threshold == 0.75

        s = buf2.sample(1, rng=random.Random(0))
        assert s[0].metadata.get("step") is not None, "Metadata lost during persistence"

        logger.info("  PASS: persistence save/load roundtrip")


def test_should_add() -> None:
    """Test the should_add threshold logic."""
    from utils.replay_buffer import ReplayBuffer

    with tempfile.TemporaryDirectory() as tmpdir:
        buf = ReplayBuffer(os.path.join(tmpdir, "buf.json"), hard_threshold=0.8)
        assert buf.should_add(0.9) is True, "difficulty=0.9 should qualify"
        assert buf.should_add(0.8) is True, "difficulty=0.8 should qualify (boundary)"
        assert buf.should_add(0.5) is False, "difficulty=0.5 should not qualify"
        assert buf.should_add(0.0) is False, "difficulty=0.0 should not qualify"

        logger.info("  PASS: should_add threshold")


def test_mix_replay_into_batch() -> None:
    """Test batch mixing with replay samples."""
    from utils.replay_buffer import ReplayBuffer, mix_replay_into_batch

    with tempfile.TemporaryDirectory() as tmpdir:
        buf = ReplayBuffer(os.path.join(tmpdir, "buf.json"), max_size=100)

        for i in range(10):
            buf.add(prompt=f"replay_p{i}", completion=f"replay_c{i}", difficulty_score=0.9)

        batch_p = [f"batch_p{i}" for i in range(20)]
        batch_c = [f"batch_c{i}" for i in range(20)]

        mixed_p, mixed_c, n_replaced = mix_replay_into_batch(
            batch_p, batch_c, buf, replay_ratio=0.1, rng=random.Random(42),
        )

        assert len(mixed_p) == 20
        assert len(mixed_c) == 20
        assert n_replaced > 0, "Expected at least 1 replacement"
        assert n_replaced <= 2, f"Expected 1-2 replacements for 10% of 20, got {n_replaced}"

        replay_count = sum(1 for p in mixed_p if p.startswith("replay_"))
        assert replay_count == n_replaced

        logger.info("  PASS: mix_replay_into_batch (%d replaced)", n_replaced)


def test_empty_buffer_mix() -> None:
    """Test that mixing with an empty buffer is a no-op."""
    from utils.replay_buffer import ReplayBuffer, mix_replay_into_batch

    with tempfile.TemporaryDirectory() as tmpdir:
        buf = ReplayBuffer(os.path.join(tmpdir, "buf.json"))

        batch_p = ["p0", "p1", "p2"]
        batch_c = ["c0", "c1", "c2"]

        mixed_p, mixed_c, n = mix_replay_into_batch(batch_p, batch_c, buf, replay_ratio=0.5)
        assert mixed_p == batch_p
        assert mixed_c == batch_c
        assert n == 0

        logger.info("  PASS: empty buffer mix is no-op")


def run_cpu_tests() -> None:
    logger.info("=" * 60)
    logger.info("Part 1: CPU unit tests")
    logger.info("=" * 60)

    test_basic_add_and_sample()
    test_fifo_eviction()
    test_persistence()
    test_should_add()
    test_mix_replay_into_batch()
    test_empty_buffer_mix()

    logger.info("")
    logger.info("All CPU tests PASSED.")


# ======================================================================
# Part 2: Optional GPU end-to-end
# ======================================================================

def run_gpu_test(args: argparse.Namespace) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("Part 2: GPU end-to-end replay buffer test")
    logger.info("=" * 60)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.makedirs(args.output_dir, exist_ok=True)

    from utils.difficulty import run_difficulty_estimation
    from utils.replay_buffer import ReplayBuffer, mix_replay_into_batch

    # Step 1: Difficulty estimation
    logger.info("Step 1: Running difficulty estimation...")
    report = run_difficulty_estimation(
        model_path=args.model_path,
        dataset_key=args.dataset_key,
        method="mv",
        n_buckets=3,
        num_generations=args.num_generations,
        max_samples=args.num_problems if args.num_problems > 0 else None,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_length=args.max_completion_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        seed=args.seed,
    )

    # Step 2: Populate buffer with Hard-bucket breakthroughs
    buf_path = os.path.join(args.output_dir, "replay_buffer.json")
    buf = ReplayBuffer(buf_path, max_size=1000, hard_threshold=0.8)

    n_added = 0
    for r in report.results:
        if buf.should_add(r.difficulty_score):
            buf.add(
                prompt=f"prompt_idx_{r.prompt_index}",
                completion=f"(synthetic breakthrough for prompt {r.prompt_index})",
                prompt_id=str(r.prompt_index),
                difficulty_score=r.difficulty_score,
                reward=1.0,
                metadata={"bucket": r.bucket_label},
            )
            n_added += 1

    buf.save()
    logger.info("Added %d hard-bucket entries to buffer (total=%d)", n_added, len(buf))

    # Step 3: Demonstrate batch mixing
    batch_p = [f"train_prompt_{i}" for i in range(16)]
    batch_c = [f"train_completion_{i}" for i in range(16)]

    mixed_p, mixed_c, n_replaced = mix_replay_into_batch(
        batch_p, batch_c, buf, replay_ratio=0.1, rng=random.Random(args.seed),
    )
    logger.info(
        "Batch mixing: %d/%d positions replaced with buffer samples",
        n_replaced, len(batch_p),
    )

    # Save summary
    summary = {
        "model_path": args.model_path,
        "dataset_key": args.dataset_key,
        "num_problems": args.num_problems,
        "difficulty_method": "mv",
        "n_buckets": 3,
        "hard_threshold": buf.hard_threshold,
        "buffer_entries_added": n_added,
        "buffer_total_size": len(buf),
        "batch_size": len(batch_p),
        "replay_ratio": 0.1,
        "n_replaced": n_replaced,
        "bucket_summary": {
            str(bid): {
                "label": report.bucket_labels.get(bid, ""),
                "count": len(indices),
            }
            for bid, indices in report.bucket_assignments.items()
        },
    }
    out_path = os.path.join(args.output_dir, "replay_test_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("GPU test summary saved to: %s", out_path)


def main() -> None:
    args = parse_args()

    run_cpu_tests()

    if args.model_path:
        run_gpu_test(args)
    else:
        logger.info("")
        logger.info("Skipping GPU end-to-end test (no --model_path provided).")
        logger.info("To run: python -m ablation.test_replay_buffer --model_path /path/to/model")


if __name__ == "__main__":
    main()
