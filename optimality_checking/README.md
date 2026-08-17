# Optimality Checking (Google OR-Tools CP-SAT)

Independent exact/bounding solver for the same exam-timetabling instances
used by `../multi_state_qubo_V2/QAOA/run_exam_scheduler.py`, so the
QUBO+SA(+repair) results can be checked against provable optimality (or,
at large scale, a provable *bound* and gap%) instead of only "zero
conflicts found."

## Why hard vs. soft constraints differ from the QUBO model

The QUBO formulation encodes C1/C2/C4 as penalty terms because QUBO has no
native way to express hard constraints. CP-SAT does, so this tool uses the
constraints as they actually are:

| Constraint | Meaning | Treatment here |
|---|---|---|
| C1 | one exam, one slot | hard (trivial) |
| C2 | conflicting exams differ | **hard** |
| C4 | per-slot seat capacity | **hard** (native `<=`, no slack bits needed) |
| C3 | same-year exams avoid consecutive slots | **soft** — minimized |

The reported objective is therefore a clean, checkable number: *the
minimum possible number of C3 (consecutive-slot) violations, subject to
zero conflicts and capacity being respected*. That is what to compare
against the SA+repair pipeline's own C3 violation count on the same
instance.

## Files

- `dataset_builder.py` — CSV loading/filtering and conflict-graph
  construction. Mirrors `run_exam_scheduler.py`'s filters exactly
  (Approved + Theory + {JUL-NOV 2025, WINTER 2025}, excluding semesters
  XI/XII) so a CP-SAT run is checking the *same* instance, not a
  look-alike one. If those filters change upstream, update this file.
- `cp_sat_optimizer.py` — the CP-SAT model, solving, and an independent
  `validate_assignment()` that recomputes C1-C4 violation counts for any
  course_code→slot assignment (CP-SAT's own, or the pipeline's).
- `check_optimality.py` — CLI entry point.
- `data/` — copies of the two enrollment CSVs behind the manuscript's
  1,171-exam Regular and 1,059-exam Makeup instances.

## Scale honestly: what CP-SAT can and can't prove here

UETP is NP-hard, and your real conflict graphs are dense (thousands of
edges). CP-SAT will:

- **Reliably prove optimality** on small subsets (tens of exams) —
  useful as a sanity check.
- **Find good solutions + a lower bound** on medium subsets, with a
  shrinking gap% as you give it more time.
- **Very likely NOT reach proven OPTIMAL** on the full 1,171/1,059-exam
  instances within a practical time budget. Report the best bound and
  gap% honestly (`is_proven_optimal: false`), the same way MIP/CP papers
  report "solved to within X% of optimality" without full closure.

Recommended workflow — scale up gradually and watch the gap:

```bash
python check_optimality.py --dataset regular --num-exams 30  --time-limit-s 60
python check_optimality.py --dataset regular --num-exams 100 --time-limit-s 600
python check_optimality.py --dataset regular --num-exams 300 --time-limit-s 1800
python check_optimality.py --dataset regular --num-exams 1000 --time-limit-s 10800   # hours
python check_optimality.py --dataset regular --k 18 --capacity 5500 --time-limit-s 14400   # full dataset, no --num-exams
```

Use `--sample-mode densest` to stress-test with the highest-conflict
subset of a given size instead of a random one (`--sample-mode random`
is the default and is representative; `--sample-mode first` is
deterministic/simplest).

## Comparing against a pipeline result

Point `--compare-timetable` at any `timetable_neal.csv`-style output from
the QUBO/SA pipeline (columns `course_code`, `time_slot`). The tool
restricts that timetable to the same subset of exams CP-SAT solved and
recomputes its C1-C4 violation counts independently, so you get a direct,
apples-to-apples comparison:

```bash
python check_optimality.py --dataset regular --num-exams 300 --time-limit-s 1800 \
  --compare-timetable "../multi_state_qubo_V2/QAOA/output/run_20260402_113602/all/timetable_neal.csv"
```

## Output

Each run writes to `output/run_<timestamp>/`:
- `result.json` — full config, status, objective/bound/gap, graph stats,
  and (if used) the pipeline comparison.
- `cp_sat_timetable.csv` — CP-SAT's own best assignment (course_code,
  time_slot).

## Install

```bash
pip install -r requirements.txt
```
