"""Uniform random demo sampler — baseline / control group.

Samples ``episodes_total`` demos uniformly at random (no importance sampling)
from the full LIBERO dataset across all suites. Every demo has equal
probability of being selected.

Usage::

    python adversarial_training/continual_learning/collect_buffer_random.py \
        --libero_dataset_dir /mnt/hlx/SimpleVLA_libero/datasets/metas \
        --output_dir /mnt/hlx/SimpleVLA_libero_data/datasets/bc_buffer_random \
        --episodes_total 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_THIS = Path(__file__).resolve()
_CODE_ROOT = _THIS.parents[2]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from adversarial_training.continual_learning.collect_buffer_bc import (
    _collect_all_demos,
    _copy_demo,
    build_meta_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Uniform random demo sampling (baseline/control)")
    p.add_argument("--libero_dataset_dir", type=str,
                   default="/mnt/hlx/SimpleVLA_libero/datasets/metas")
    p.add_argument("--output_dir", type=str,
                   default="/mnt/hlx/SimpleVLA_libero_data/datasets/bc_buffer_random")
    p.add_argument("--episodes_total", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--suites", nargs="+",
                   default=["libero_10", "libero_goal",
                            "libero_object", "libero_spatial"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    suites = list(args.suites)
    n_total = args.episodes_total
    rng = np.random.default_rng(args.seed)

    print(f"[collect_buffer_random] Uniform random sampling")
    print(f"  source={args.libero_dataset_dir}  suites={suites}")
    print(f"  episodes_total={n_total}  seed={args.seed}")
    print(f"  output → {out_root}")

    # ── Collect all demos ──
    entries = _collect_all_demos(args.libero_dataset_dir, suites, dry_run=False)
    M = len(entries)
    print(f"[collect_buffer_random] {M} total demos across {len(suites)} suites")

    # ── Uniform random sampling (without replacement) ──
    if n_total > M:
        print(f"[collect_buffer_random] n_total={n_total} > {M} demos, "
              f"sampling with replacement")
        replace = True
    else:
        replace = False
    draws = rng.choice(M, size=n_total, replace=replace)
    print(f"[collect_buffer_random] {n_total} draws  "
          f"replace={replace}  unique={len(np.unique(draws))}")

    per_suite: Dict[str, int] = {}
    per_task: Dict[Tuple[str, int], int] = {}
    for idx in draws:
        e = entries[idx]
        per_suite[e.suite_name] = per_suite.get(e.suite_name, 0) + 1
        per_task[(e.suite_name, e.task_index)] = \
            per_task.get((e.suite_name, e.task_index), 0) + 1

    for s in suites:
        n_s = per_suite.get(s, 0)
        t_with = sum(1 for (sn, _), c in per_task.items()
                     if sn == s and c > 0)
        print(f"  {s}: {n_s} draws  ({t_with} tasks)")

    # ── Copy demos (all with uniform weight = 1.0) ──
    demo_counter: Dict[Tuple[str, int], int] = {}
    for draw_idx in draws:
        e = entries[draw_idx]
        key = (e.suite_name, e.task_index)
        di = demo_counter.get(key, 0)
        demo_counter[key] = di + 1
        _copy_demo(e, out_root, demo_index=di, is_weight=1.0)

    total_files = sum(demo_counter.values())
    print(f"[collect_buffer_random] wrote {total_files} HDF5 files")

    # ── Metadata ──
    meta = build_meta_json(out_root, suites)
    meta_path = out_root / "bc_train_meta.json"
    import json
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[collect_buffer_random] metadata → {meta_path}  "
          f"({meta['num_episodes']} demos)")


if __name__ == "__main__":
    main()
