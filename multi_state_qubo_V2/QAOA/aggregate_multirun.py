"""
Aggregate statistics across multiple independent SA runs of the same
configuration, to answer "is this result reliable, or a lucky one-off?"

Usage (run on the server, from multi_state_qubo_V2/QAOA/):
    python3 aggregate_multirun.py --output-dir output --since "20260101_000000"

Scans output/run_*/all/neal_results.json (or output/run_*/neal_results.json
depending on --adjacency-mode both vs single-mode layout), extracts the key
metrics, and reports mean/std/min/max/best across all matching runs.
"""
import argparse
import glob
import json
import statistics as stats
from pathlib import Path


def load_run(path):
    with open(path) as f:
        d = json.load(f)
    repair = d.get('repair', {}) or {}
    se = repair.get('slot_expansion', {}) or {}
    return {
        'run': Path(path).parts[-3],  # output/run_XXXXX/all/neal_results.json -> run_XXXXX
        'energy': d.get('energy'),
        'runtime_s': d.get('runtime_seconds'),
        'c1': d.get('c1_onehot_violations'),
        'c2': d.get('c2_conflict_violations'),
        'c3': d.get('c3_consecutive_violations'),
        'c4': d.get('c4_slots_over_capacity'),
        'is_valid': d.get('is_valid'),
        'repair_applied': repair.get('applied'),
        'repaired_exam_count': se.get('repaired_exam_count'),
        'k_repair': se.get('k_repair'),
        'new_slots_added': se.get('new_slots_added'),
        'total_slots_final': se.get('total_slots_final'),
        'colors_used': d.get('colors_used'),
    }


def summarize(values, label):
    vals = [v for v in values if v is not None]
    if not vals:
        print(f"  {label}: no data")
        return
    print(f"  {label}: mean={stats.mean(vals):.2f}  "
          f"std={stats.pstdev(vals):.2f}  "
          f"min={min(vals)}  max={max(vals)}  n={len(vals)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='output')
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.output_dir}/run_*/all/neal_results.json")) \
        or sorted(glob.glob(f"{args.output_dir}/run_*/neal_results.json"))

    if not paths:
        print(f"No neal_results.json found under {args.output_dir}/run_*/")
        return

    runs = [load_run(p) for p in paths]

    print(f"Found {len(runs)} runs:\n")
    for r in runs:
        print(f"  {r['run']}: energy={r['energy']:.2f}  valid={r['is_valid']}  "
              f"C1/C2/C3/C4={r['c1']}/{r['c2']}/{r['c3']}/{r['c4']}  "
              f"repair_applied={r['repair_applied']}  k_repair={r['k_repair']}  "
              f"total_slots={r['total_slots_final']}  colors_used={r['colors_used']}")

    print("\n=== Aggregate statistics across runs ===")
    summarize([r['energy'] for r in runs], 'Energy')
    summarize([r['runtime_s'] for r in runs], 'Runtime (s)')
    summarize([r['c1'] for r in runs], 'C1 violations (post-repair)')
    summarize([r['c2'] for r in runs], 'C2 violations (post-repair)')
    summarize([r['c3'] for r in runs], 'C3 violations (post-repair)')
    summarize([r['c4'] for r in runs], 'C4 violations (post-repair)')
    summarize([r['k_repair'] for r in runs], 'k_repair (repair colours needed)')
    summarize([r['total_slots_final'] for r in runs], 'Total slots after repair')
    summarize([r['colors_used'] for r in runs], 'Colours (slots) actually used')

    n_valid = sum(1 for r in runs if r['is_valid'])
    print(f"\n  Fully valid (C1=C2=C3=C4=0) in {n_valid}/{len(runs)} runs")


if __name__ == '__main__':
    main()
