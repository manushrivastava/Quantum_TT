"""
Diagnose WHY a pipeline run's repair procedure used more slots than the
global true minimum (found by find_min_slots.py / verify against the
whole graph).

The repair procedure freezes the original SA assignment for every
non-violated exam and only re-slots the small repair set, placing each
local colour-group into an existing frozen slot if compatible, or
appending a new slot otherwise. This script asks the precise question
that isolates the cause of any gap: GIVEN the exact frozen background
from the real run, what is the TRUE MINIMUM number of NEW slots needed
to place the repair-set exams -- respecting C2/C3 against the frozen
exams and C4 against the frozen slot loads -- versus what the pipeline's
own placement search actually used?

- If CP-SAT's minimum equals what the pipeline used: the placement step
  already found the true optimum GIVEN the frozen background; any
  remaining gap to the global minimum is structural (only achievable by
  also re-touching the frozen exams, i.e. a fundamentally different,
  jointly-optimised assignment).
- If CP-SAT's minimum is LOWER than what the pipeline used: the
  placement search itself left slots on the table -- a fixable
  engineering gap in the repair procedure, not a structural limit.

Usage:
    python diagnose_repair_gap.py \\
        --run-dir "../multi_state_qubo_V2/QAOA/output/run_20260402_113538/all" \\
        --timetable-csv "../multi_state_qubo_V2/QAOA/output/run_20260402_113538/all/timetable_neal.csv" \\
        --original-k 18 --capacity 5500
"""

import argparse

import pandas as pd
from ortools.sat.python import cp_model

from dataset_builder import load_saved_run_instance


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--timetable-csv', required=True)
    p.add_argument('--original-k', type=int, required=True)
    p.add_argument('--capacity', type=int, required=True)
    p.add_argument('--time-limit-s', type=float, default=300)
    p.add_argument('--num-workers', type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    courses_df, adjacency = load_saved_run_instance(args.run_dir)
    timetable = pd.read_csv(args.timetable_csv)

    code_to_idx = {row['course_code']: i for i, row in courses_df.iterrows()}
    enrollment = courses_df['enrollment'].tolist()
    year = courses_df['year'].tolist()

    frozen = timetable[timetable['time_slot'] < args.original_k]
    repair_rows = timetable[timetable['time_slot'] >= args.original_k]
    repair_codes = repair_rows['course_code'].tolist()
    actual_new_slots_used = repair_rows['time_slot'].nunique()

    print("=" * 70)
    print("REPAIR-PLACEMENT GAP DIAGNOSTIC")
    print("=" * 70)
    print(f"Frozen (non-repaired) exams: {len(frozen)}  in slots 0..{args.original_k - 1}")
    print(f"Repair-set exams: {len(repair_codes)}  {repair_codes}")
    print(f"Pipeline actually used {actual_new_slots_used} NEW slots for the repair set "
          f"(slots {sorted(repair_rows['time_slot'].unique())})")

    # Frozen slot contents: idx -> slot, and per-slot frozen load
    frozen_slot_of = {}
    frozen_load = {k: 0 for k in range(args.original_k)}
    for _, row in frozen.iterrows():
        idx = code_to_idx[row['course_code']]
        k = int(row['time_slot'])
        frozen_slot_of[idx] = k
        frozen_load[k] += enrollment[idx]

    repair_idx = [code_to_idx[c] for c in repair_codes]
    R = len(repair_idx)
    K_orig = args.original_k

    def feasible_with_m_new_slots(m, buffer=0):
        """Feasibility check: can the repair set be placed using the frozen
        background plus exactly m CONTIGUOUS new slots, placed after an
        optional `buffer` number of empty slots right after slot K_orig-1?
        (K_orig+buffer .. K_orig+buffer+m-1). A buffer costs nothing in a
        real compacted schedule (an empty slot is simply never numbered),
        but it DOES matter for C3 adjacency -- an empty gap slot has no
        exam to conflict with, so it can break an otherwise-forced
        adjacency between the last frozen slot and the first new one. The
        real pipeline run being diagnosed here does exactly this (slot 18
        stayed empty; repair exams used 19-23).

        Using a contiguous range -- rather than letting the solver pick
        slot indices freely from a larger pool and only counting distinct
        ones used -- is essential: a sparse/non-contiguous choice could
        dodge C3 by leaving gaps a real compacted schedule couldn't have.
        """
        new_start = K_orig + buffer
        all_slots = list(range(K_orig)) + list(range(new_start, new_start + m))

        model = cp_model.CpModel()
        y = {}
        for i in repair_idx:
            for s in all_slots:
                forbidden = False
                if s < K_orig:
                    for j, sj in frozen_slot_of.items():
                        if sj == s and adjacency[i, j]:
                            forbidden = True
                            break
                if not forbidden:
                    y[(i, s)] = model.NewBoolVar(f'y_{i}_{s}')

        for i in repair_idx:
            opts = [y[(i, s)] for s in all_slots if (i, s) in y]
            if not opts:
                return 'INFEASIBLE', 0.0  # no legal slot at all for this exam
            model.AddExactlyOne(opts)

        for a in range(len(repair_idx)):
            for b in range(a + 1, len(repair_idx)):
                i, j = repair_idx[a], repair_idx[b]
                if adjacency[i, j]:
                    for s in all_slots:
                        if (i, s) in y and (j, s) in y:
                            model.AddAtMostOne([y[(i, s)], y[(j, s)]])

        # C3 vs frozen neighbours in adjacent EXISTING slots
        for i in repair_idx:
            for s in all_slots:
                if (i, s) not in y:
                    continue
                for neighbour_slot in (s - 1, s + 1):
                    if 0 <= neighbour_slot < K_orig:
                        for j, sj in frozen_slot_of.items():
                            if sj == neighbour_slot and adjacency[i, j] and year[i] == year[j]:
                                model.Add(y[(i, s)] == 0)
                                break

        # C3 among repair exams in truly-adjacent (contiguous) slots
        for a in range(len(repair_idx)):
            for b in range(len(repair_idx)):
                if a == b:
                    continue
                i, j = repair_idx[a], repair_idx[b]
                if adjacency[i, j] and year[i] == year[j]:
                    for s in all_slots:
                        if (i, s) in y and (j, s + 1) in y:
                            model.AddAtMostOne([y[(i, s)], y[(j, s + 1)]])

        for s in all_slots:
            base_load = frozen_load.get(s, 0)
            terms = [enrollment[i] * y[(i, s)] for i in repair_idx if (i, s) in y]
            if terms:
                model.Add(base_load + sum(terms) <= args.capacity)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = args.time_limit_s
        solver.parameters.num_search_workers = args.num_workers
        status = solver.Solve(model)
        return solver.StatusName(status), 0.0

    print("-" * 70)
    print(f"Searching smallest m (new slots used), trying buffer offsets 0, 1, 2 "
          f"empty slots after slot {K_orig - 1} ...")
    best = None  # (m, buffer)
    for buffer in (0, 1, 2):
        for m in range(1, R + 1):
            status_name = feasible_with_m_new_slots(m, buffer)[0]
            print(f"  buffer={buffer}, m={m}: {status_name}")
            if status_name in ('FEASIBLE', 'OPTIMAL'):
                if best is None or m < best[0]:
                    best = (m, buffer)
                break  # smallest feasible m for this buffer found; try next buffer

    print("-" * 70)
    if best is not None:
        min_new, best_buffer = best
        print(f"TRUE MINIMUM new slots USED, GIVEN this exact frozen background: {min_new} "
              f"(with {best_buffer} empty buffer slot(s) after slot {K_orig - 1})")
        print(f"Pipeline's actual placement search used: {actual_new_slots_used}")
        if min_new < actual_new_slots_used:
            print(f"=> Placement search left slack on the table: {actual_new_slots_used - min_new} "
                  f"fewer new slots were achievable even WITHOUT touching the frozen exams.")
        elif min_new == actual_new_slots_used:
            print("=> Placement search was already optimal given the frozen background. "
                  "Any remaining gap to the global minimum requires re-optimising the "
                  "frozen exams too (a structural limit of the freeze-and-patch design).")
        else:
            print("=> WARNING: pipeline used FEWER new slots than this check found feasible -- "
                  "investigate model discrepancy before trusting either number.")
    else:
        print(f"No feasible (m, buffer) found within the tested range/time budget.")


if __name__ == '__main__':
    main()
