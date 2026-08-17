"""
Find the minimum number of time slots (colours) needed to satisfy
C1 (one-hot), C2 (conflict avoidance), and C4 (capacity) -- ignoring C3
entirely. This is the graph-colouring "chromatic number" question (with
capacity folded in), distinct from "how many of a given K slots were
used" (which check_optimality.py already reports per run).

Directly answers: could the pipeline's repair procedure have used fewer
than the 24 slots it settled on for the full-scale schedule? Feeds the
manuscript's own "Adaptive K Estimation" future-work discussion with an
actual number (or a certified bound) instead of leaving it open.

Usage:
    python find_min_slots.py --dataset regular --num-exams 100 --capacity 5500
    python find_min_slots.py --dataset regular --num-exams 100 --capacity 5500 --k-max 30
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from dataset_builder import build_courses_and_adjacency, graph_summary, load_and_filter_csv, select_subset
from cp_sat_optimizer import find_minimum_k

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'

BUNDLED_DATASETS = {
    'regular': DATA_DIR / 'Student Course (Jul-Nov 2025 and Winter 2025) Regular Elective.csv',
    'makeup': DATA_DIR / 'Student Course (Jul-Nov 2025 and Winter 2025) Makeup Elective.csv',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--dataset', choices=sorted(BUNDLED_DATASETS.keys()))
    src.add_argument('--input-csv', type=str)

    p.add_argument('--adjacency-mode', choices=['major', 'all'], default='all')
    p.add_argument('--max-rows', type=int, default=None)
    p.add_argument('--num-exams', type=int, default=None)
    p.add_argument('--sample-mode', choices=['random', 'densest', 'first'], default='random')
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--capacity', type=int, default=5500)
    p.add_argument('--k-min', type=int, default=1)
    p.add_argument('--k-max', type=int, default=None,
                    help="Upper bound to sanity-check before binary search (default: num exams).")
    p.add_argument('--time-limit-per-k-s', type=float, default=60,
                    help="Per-K feasibility-check time budget (default: 60s).")
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--output-dir', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    input_csv = str(BUNDLED_DATASETS[args.dataset]) if args.dataset else args.input_csv

    print("=" * 70)
    print("MINIMUM-SLOTS (CHROMATIC NUMBER + CAPACITY) CHECK")
    print("=" * 70)

    filtered = load_and_filter_csv(input_csv, max_rows=args.max_rows)
    courses_df, adjacency = build_courses_and_adjacency(filtered, adjacency_mode=args.adjacency_mode)
    sub_courses_df, sub_adjacency = select_subset(
        courses_df, adjacency, num_exams=args.num_exams, sample_mode=args.sample_mode, seed=args.seed,
    )
    summary = graph_summary(sub_courses_df, sub_adjacency)
    print(f"Instance: {summary['num_courses']} exams, {summary['num_conflict_edges']} conflict edges "
          f"({summary['density_pct']}% density), capacity={args.capacity}")
    print(f"Binary-searching K in [{args.k_min}, {args.k_max or summary['num_courses']}] "
          f"({args.time_limit_per_k_s}s per K) ...")
    print("-" * 70)

    result = find_minimum_k(
        sub_courses_df, sub_adjacency, capacity=args.capacity,
        k_min=args.k_min, k_max=args.k_max,
        time_limit_per_k_s=args.time_limit_per_k_s, num_workers=args.num_workers,
    )

    print("-" * 70)
    if result['min_k_found'] is not None:
        closed = result.get('search_closed', False)
        print(f"Smallest FEASIBLE K found : {result['min_k_found']}"
              f"{'  (PROVEN MINIMUM)' if closed else '  (not yet proven minimal -- see certified_infeasible_up_to)'}")
        print(f"Certified infeasible up to: K <= {result['certified_infeasible_up_to']}")
    else:
        print(f"Message: {result['message']}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        OUTPUT_DIR / f"minslots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'min_slots_result.json', 'w') as f:
        json.dump({'config': vars(args), 'instance_summary': summary, 'result': result}, f, indent=2, default=str)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
