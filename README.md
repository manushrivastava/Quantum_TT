# Quantum_TT

QUBO-based examination timetabling: exam scheduling formulated as a
Quadratic Unconstrained Binary Optimization problem (C1 one-hot, C2
conflict avoidance, C3 consecutive-slot avoidance, C4 capacity),
solved via classical Simulated Annealing (D-Wave Neal) with SPSA-based
automated penalty tuning and a hybrid local-repair procedure.

## Structure

- `multi_state_qubo_V2/` — main QUBO construction, SA solving, SPSA
  penalty tuner, and hybrid repair pipeline (`QAOA/run_exam_scheduler.py`).
- `optimality_checking/` — independent optimality verification using
  Google OR-Tools CP-SAT (exact solver): proves minimality of the
  repair procedure's local colour count, finds true minimum slot
  counts, and diagnoses repair-placement gaps.

## Data is intentionally not included in this repository

The enrolment CSVs used to build the conflict graphs (student
registration data) are institutional records and are **not** published
here. To run either pipeline you need to supply your own enrolment CSV
(see `multi_state_qubo_V2/QAOA/run_exam_scheduler.md` for the expected
schema and filtering rules) or transfer the source data separately via
a private channel (scp/rsync), not through this repository.

## Environment

See `multi_state_qubo_V2/reqirements.txt` (or `pyproject.toml`) and
`optimality_checking/requirements.txt`.
