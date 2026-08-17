"""
Certify (or refute) that k_rep, as found by the QUBO/SA pipeline's own
hybrid repair procedure, is truly the MINIMUM number of local colours
needed for the actual repair set from a real run -- not just the
smallest k that Neal's heuristic sampling happened to succeed at.

The pipeline's repair procedure tries k=1,2,3,... and accepts the first
k for which Neal finds a sample with C1=C2=C3=C4=0. That is an empirical
"first success", not a proof that smaller k are impossible -- Neal could
simply have failed to find a feasible sample at a smaller k within its
sampling budget. This script re-checks the exact same repair subgraph
with CP-SAT, which either PROVES infeasibility at k < k_rep (confirming
Neal's result was not just bad luck) or finds a smaller feasible k
(meaning k_rep was larger than necessary).

The repair-set identity is recovered directly from a saved pipeline run:
any exam placed in a slot >= the original --k is, by construction, part
of a repair group that could not be placed into an existing frozen slot.
Combined with the run's own saved courses.csv + conflict_adjacency.csv,
this reconstructs the *exact* sub-QUBO instance the repair procedure
solved -- not a look-alike sample of the same size.

Usage:
    python verify_repair_set.py \\
        --run-dir "../multi_state_qubo_V2/QAOA/output/run_20260402_113602/all" \\
        --timetable-csv "../multi_state_qubo_V2/QAOA/output/run_20260402_113602/all/timetable_neal.csv" \\
        --original-k 18 --capacity 5500 --reported-k-rep 5
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from dataset_builder import graph_summary, load_saved_run_instance, select_by_course_codes
from cp_sat_optimizer import find_minimum_k

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'output'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True,
                    help="Directory with courses.csv + conflict_adjacency.csv from a saved pipeline run.")
    p.add_argument('--timetable-csv', required=True,
                    help="That run's final timetable_neal.csv (course_code, time_slot, ...).")
    p.add_argument('--original-k', type=int, required=True,
                    help="The --k the pipeline was originally run with (e.g. 18). Any exam in a "
                         "slot >= this value is, by construction, part of a repair group.")
    p.add_argument('--capacity', type=int, required=True)
    p.add_argument('--reported-k-rep', type=int, default=None,
                    help="k_rep the paper reports for this run (for reference in the printed summary).")
    p.add_argument('--k-max', type=int, default=None,
                    help="Upper bound for the search (default: repair-set size).")
    p.add_argument('--time-limit-per-k-s', type=float, default=120)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--output-dir', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("REPAIR-SET MINIMUM-k CERTIFICATION")
    print("=" * 70)

    courses_df, adjacency = load_saved_run_instance(args.run_dir)
    timetable = pd.read_csv(args.timetable_csv)

    repair_codes = timetable.loc[timetable['time_slot'] >= args.original_k, 'course_code'].tolist()
    print(f"Recovered repair set from {args.timetable_csv}:")
    print(f"  {len(repair_codes)} exams placed in slots >= {args.original_k}: {repair_codes}")

    sub_courses_df, sub_adjacency = select_by_course_codes(courses_df, adjacency, repair_codes)
    summary = graph_summary(sub_courses_df, sub_adjacency)
    print(f"Repair-subgraph: {summary['num_courses']} exams, {summary['num_conflict_edges']} conflict "
          f"edges ({summary['density_pct']}% density)")
    if args.reported_k_rep is not None:
        print(f"Paper-reported k_rep (found via Neal, empirical first-success): {args.reported_k_rep}")

    k_max = args.k_max or summary['num_courses']
    print(f"Certifying minimum k with C1=C2=C3=C4=0 ALL enforced as hard "
          f"(k in [1, {k_max}], {args.time_limit_per_k_s}s per K) ...")
    print("-" * 70)

    result = find_minimum_k(
        sub_courses_df, sub_adjacency, capacity=args.capacity,
        k_min=1, k_max=k_max, time_limit_per_k_s=args.time_limit_per_k_s,
        num_workers=args.num_workers, enforce_c3=True,
    )

    print("-" * 70)
    if result['min_k_found'] is not None:
        closed = result.get('search_closed', False)
        print(f"CP-SAT certified minimum k : {result['min_k_found']}"
              f"{'  (PROVEN MINIMUM)' if closed else '  (not fully closed -- see log)'}")
        print(f"Certified infeasible up to : k <= {result['certified_infeasible_up_to']}")
        if args.reported_k_rep is not None:
            if result['min_k_found'] == args.reported_k_rep and closed:
                print(f"=> CONFIRMED: the paper's k_rep={args.reported_k_rep} is the PROVEN minimum, "
                      f"not just Neal's first empirical success.")
            elif result['min_k_found'] < args.reported_k_rep:
                print(f"=> The paper's k_rep={args.reported_k_rep} was NOT minimal: CP-SAT found a "
                      f"feasible k={result['min_k_found']} < {args.reported_k_rep}.")
            elif not closed:
                print(f"=> Inconclusive within the time budget: cannot yet confirm k_rep="
                      f"{args.reported_k_rep} is minimal. Increase --time-limit-per-k-s.")
    else:
        print(f"Message: {result['message']}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        OUTPUT_DIR / f"repaircheck_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'repair_set_result.json', 'w') as f:
        json.dump({
            'config': vars(args),
            'repair_codes': repair_codes,
            'instance_summary': summary,
            'result': result,
        }, f, indent=2, default=str)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
