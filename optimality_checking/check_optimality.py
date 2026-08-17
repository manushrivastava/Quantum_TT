"""
Check how close the QUBO+SA(+repair) pipeline's exam timetable is to
provable optimality, using Google OR-Tools CP-SAT as an independent
exact/bounding solver.

Usage examples
--------------
Smoke test on a small random subset (should reach proven OPTIMAL quickly):

    python check_optimality.py --dataset regular --num-exams 30 --time-limit-s 60

Scale up gradually to see how the optimality gap grows with size:

    python check_optimality.py --dataset regular --num-exams 100 --time-limit-s 600
    python check_optimality.py --dataset regular --num-exams 300 --time-limit-s 1800

Full dataset (matches the manuscript's instance: 1,171 Regular / 1,059
Makeup exams at K=18, capacity=5500). This will very likely NOT reach
proven OPTIMAL -- report the best bound and gap%, not "optimal":

    python check_optimality.py --dataset regular --k 18 --capacity 5500 \\
        --time-limit-s 14400

Compare CP-SAT's result against a timetable already produced by the
QUBO/SA pipeline (e.g. output/run_.../all/timetable_neal.csv) on the
same subset:

    python check_optimality.py --dataset regular --num-exams 100 \\
        --compare-timetable ../multi_state_qubo_V2/QAOA/output/run_.../all/timetable_neal.csv
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from dataset_builder import (
    build_courses_and_adjacency,
    graph_summary,
    load_and_filter_csv,
    select_subset,
)
from cp_sat_optimizer import build_and_solve, validate_assignment

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'

BUNDLED_DATASETS = {
    'regular': DATA_DIR / 'Student Course (Jul-Nov 2025 and Winter 2025) Regular Elective.csv',
    'makeup': DATA_DIR / 'Student Course (Jul-Nov 2025 and Winter 2025) Makeup Elective.csv',
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Check optimality of exam-timetabling solutions via OR-Tools CP-SAT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--dataset', choices=sorted(BUNDLED_DATASETS.keys()),
                      help="Use one of the bundled CSVs in data/.")
    src.add_argument('--input-csv', type=str, help="Path to a custom enrollment CSV.")

    p.add_argument('--adjacency-mode', choices=['major', 'all'], default='all',
                    help="Conflict graph to use (default: all, matches the paper's runs).")
    p.add_argument('--max-rows', type=int, default=None,
                    help="Optional: only read the first N raw CSV rows before filtering "
                         "(fast smoke tests). Independent of --num-exams.")

    p.add_argument('--num-exams', type=int, default=None,
                    help="Restrict to a subset of N exams (induced subgraph). "
                         "Omit, or set >= total exams, to run on the FULL dataset.")
    p.add_argument('--sample-mode', choices=['random', 'densest', 'first'], default='random',
                    help="How to pick the subset (default: random).")
    p.add_argument('--seed', type=int, default=42, help="Random seed for --sample-mode random.")

    p.add_argument('--k', type=int, default=18, help="Number of time slots (default: 18).")
    p.add_argument('--capacity', type=int, default=5500,
                    help="Per-slot seat capacity, hard constraint (default: 5500).")

    p.add_argument('--time-limit-s', type=float, default=300,
                    help="CP-SAT wall-clock time budget in seconds (default: 300). "
                         "Full-scale runs need much larger budgets (hours) and may "
                         "still only return a bound, not a proven optimum.")
    p.add_argument('--num-workers', type=int, default=8, help="CP-SAT parallel search workers.")
    p.add_argument('--verbose', action='store_true', help="Stream CP-SAT search log.")

    p.add_argument('--compare-timetable', type=str, default=None,
                    help="Path to a timetable_neal.csv-style file (columns: "
                         "course_code, time_slot) from the QUBO/SA pipeline to "
                         "validate on the same subset and compare against CP-SAT.")

    p.add_argument('--output-dir', type=str, default=None,
                    help="Where to write results (default: output/run_<timestamp>/).")
    return p.parse_args()


def load_compare_timetable(path):
    df = pd.read_csv(path)
    col_slot = 'time_slot' if 'time_slot' in df.columns else df.columns[0]
    col_code = 'course_code' if 'course_code' in df.columns else df.columns[1]
    return dict(zip(df[col_code], df[col_slot]))


def main():
    args = parse_args()

    input_csv = str(BUNDLED_DATASETS[args.dataset]) if args.dataset else args.input_csv

    print("=" * 70)
    print("OPTIMALITY CHECK (Google OR-Tools CP-SAT)")
    print("=" * 70)
    print(f"Input CSV       : {input_csv}")
    print(f"Adjacency mode  : {args.adjacency_mode}")

    filtered = load_and_filter_csv(input_csv, max_rows=args.max_rows)
    courses_df, adjacency = build_courses_and_adjacency(filtered, adjacency_mode=args.adjacency_mode)
    full_summary = graph_summary(courses_df, adjacency)
    print(f"Full instance   : {full_summary['num_courses']} exams, "
          f"{full_summary['num_conflict_edges']} conflict edges "
          f"({full_summary['density_pct']}% density)")

    sub_courses_df, sub_adjacency = select_subset(
        courses_df, adjacency,
        num_exams=args.num_exams, sample_mode=args.sample_mode, seed=args.seed,
    )
    sub_summary = graph_summary(sub_courses_df, sub_adjacency)
    if args.num_exams is not None and args.num_exams < full_summary['num_courses']:
        print(f"Subset selected : {sub_summary['num_courses']} exams "
              f"(mode={args.sample_mode}, seed={args.seed}), "
              f"{sub_summary['num_conflict_edges']} conflict edges "
              f"({sub_summary['density_pct']}% density)")
    else:
        print("Subset selected : FULL dataset (no --num-exams restriction)")
        if args.time_limit_s < 3600:
            print(f"  NOTE: running on the full dataset with only "
                  f"{args.time_limit_s:.0f}s time limit. CP-SAT is very unlikely "
                  f"to reach proven optimality at this scale; expect status "
                  f"FEASIBLE/UNKNOWN with a bound + gap%, not OPTIMAL. "
                  f"Increase --time-limit-s for a tighter bound.")

    print(f"K (slots)       : {args.k}   (greedy upper-bound hint on colours needed: "
          f"{sub_summary['greedy_upper_bound_k']}; NOT a proven lower bound)")
    print(f"Capacity        : {args.capacity}")
    print("-" * 70)
    print("Solving with CP-SAT ...")

    result = build_and_solve(
        sub_courses_df, sub_adjacency,
        K=args.k, capacity=args.capacity,
        time_limit_s=args.time_limit_s, num_workers=args.num_workers,
        verbose=args.verbose,
    )

    print("-" * 70)
    print(f"Status               : {result['status']}"
          f"{'  (PROVEN OPTIMAL)' if result['is_proven_optimal'] else ''}")
    print(f"Wall time            : {result['wall_time_s']}s")
    if 'c3_violations_best' in result:
        print(f"C3 violations (best) : {result['c3_violations_best']}")
        print(f"C3 lower bound       : {result['c3_violations_lower_bound']}")
        print(f"Optimality gap       : {result['optimality_gap_pct']}%")
        recheck = validate_assignment(result['assignment'], sub_courses_df, sub_adjacency,
                                       args.k, args.capacity)
        print(f"Independent recheck  : {recheck}")
    else:
        print(f"Message              : {result.get('message', '')}")

    if args.compare_timetable:
        print("-" * 70)
        print(f"Comparing against    : {args.compare_timetable}")
        full_assignment = load_compare_timetable(args.compare_timetable)
        sub_codes = set(sub_courses_df['course_code'])
        pipeline_assignment = {c: s for c, s in full_assignment.items() if c in sub_codes}
        missing_from_compare = sub_codes - set(pipeline_assignment.keys())
        if missing_from_compare:
            print(f"  WARNING: {len(missing_from_compare)} exams in this subset are not "
                  f"present in the compare-timetable file (different run/instance?). "
                  f"Comparison is only over the {len(pipeline_assignment)} overlapping exams.")
        pipeline_metrics = validate_assignment(pipeline_assignment, sub_courses_df, sub_adjacency,
                                                args.k, args.capacity)
        print(f"  Pipeline timetable on this subset: {pipeline_metrics}")
        if 'c3_violations_best' in result:
            print(f"  CP-SAT best/bound on this subset : "
                  f"{result['c3_violations_best']} / {result['c3_violations_lower_bound']} "
                  f"(gap {result['optimality_gap_pct']}%)")
            print(f"  Pipeline C3 violations            : "
                  f"{pipeline_metrics['c3_consecutive_violations']}")
        result['pipeline_comparison'] = pipeline_metrics

    output_dir = Path(args.output_dir) if args.output_dir else (
        OUTPUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    result_to_save = {k: v for k, v in result.items() if k != 'assignment'}
    with open(output_dir / 'result.json', 'w') as f:
        json.dump({'config': run_config, 'result': result_to_save,
                   'full_instance_summary': full_summary,
                   'solved_instance_summary': sub_summary}, f, indent=2, default=str)

    if 'assignment' in result:
        pd.DataFrame(
            [{'course_code': c, 'time_slot': k} for c, k in result['assignment'].items()]
        ).to_csv(output_dir / 'cp_sat_timetable.csv', index=False)

    print("-" * 70)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
