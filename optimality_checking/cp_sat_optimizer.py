"""
CP-SAT model for the same UETP instance used by the QUBO/SA pipeline,
built for provable-optimality checking rather than penalty-weighted
heuristic search.

Constraint treatment (deliberately different from the QUBO's penalty
encoding, and closer to how the OR literature reports "solved to X%
of optimality"):

    C1 (one exam, one slot)         -> HARD, trivial via ExactlyOne
    C2 (conflicting exams differ)   -> HARD  (AddAtMostOne per edge/slot)
    C4 (slot seat-capacity)         -> HARD  (native <= constraint, no
                                        slack-bit encoding needed in CP-SAT)
    C3 (same-year consecutive-slot) -> SOFT, minimized in the objective

This means the objective value CP-SAT reports is the true minimum
achievable number of C3 (consecutive-slot) violations subject to zero
conflicts and capacity being respected -- a well-defined, checkable
quantity to compare against the SA+repair pipeline's own C3 violation
count on the same instance.
"""

import time

from ortools.sat.python import cp_model


def build_and_solve(courses_df, adjacency, K, capacity, time_limit_s=300,
                     num_workers=8, verbose=False):
    n = len(courses_df)
    enrollment = courses_df['enrollment'].tolist()
    year = courses_df['year'].tolist()

    model = cp_model.CpModel()

    # x[i][k] = 1 if exam i is assigned to slot k
    x = [[model.NewBoolVar(f'x_{i}_{k}') for k in range(K)] for i in range(n)]

    # --- C1: each exam exactly one slot (hard) ---
    for i in range(n):
        model.AddExactlyOne(x[i])

    # --- C2: conflicting exams must not share a slot (hard) ---
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adjacency[i, j]]
    for (i, j) in edges:
        for k in range(K):
            model.AddAtMostOne([x[i][k], x[j][k]])

    # --- C4: per-slot seat load must not exceed capacity (hard) ---
    for k in range(K):
        model.Add(sum(enrollment[i] * x[i][k] for i in range(n)) <= capacity)

    # --- C3: same-year conflicting pairs in consecutive slots (soft) ---
    same_year_edges = [(i, j) for (i, j) in edges if year[i] == year[j]]
    violations = []
    for (i, j) in same_year_edges:
        for k in range(K - 1):
            v1 = model.NewBoolVar(f'c3_{i}_{j}_{k}_fwd')
            model.Add(v1 >= x[i][k] + x[j][k + 1] - 1)
            v2 = model.NewBoolVar(f'c3_{i}_{j}_{k}_bwd')
            model.Add(v2 >= x[i][k + 1] + x[j][k] - 1)
            violations.extend([v1, v2])

    if violations:
        model.Minimize(sum(violations))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = num_workers
    solver.parameters.log_search_progress = verbose

    start = time.time()
    status = solver.Solve(model)
    wall_time = time.time() - start

    status_name = solver.StatusName(status)
    result = {
        'status': status_name,
        'is_proven_optimal': status == cp_model.OPTIMAL,
        'wall_time_s': round(wall_time, 2),
        'num_exams': n,
        'num_conflict_edges': len(edges),
        'num_same_year_conflict_edges': len(same_year_edges),
        'K': K,
        'capacity': capacity,
    }

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        objective = solver.ObjectiveValue()
        bound = solver.BestObjectiveBound()
        gap_pct = (
            round((objective - bound) / objective * 100, 3) if objective > 0 else 0.0
        )
        assignment = {
            courses_df.iloc[i]['course_code']: k
            for i in range(n) for k in range(K) if solver.Value(x[i][k]) == 1
        }
        result.update({
            'c3_violations_best': int(objective),
            'c3_violations_lower_bound': int(bound),
            'optimality_gap_pct': gap_pct,
            'assignment': assignment,
        })
    elif status == cp_model.INFEASIBLE:
        result['message'] = (
            'No feasible schedule exists under the given hard constraints '
            '(C1, C2, C4) with K slots and this capacity. Try increasing '
            '--k or --capacity.'
        )
    else:
        result['message'] = (
            'Solver returned UNKNOWN: no feasible solution was found within '
            'the time limit. Increase --time-limit-s, or reduce --num-exams '
            'to check optimality on a smaller instance first.'
        )

    return result


def _feasible_at_k(courses_df, adjacency, K, capacity, time_limit_s, num_workers,
                    enforce_c3=False, get_solution=False):
    """Feasibility-only check (no objective): can the hard constraints be
    satisfied with exactly K slots available? Returns (status_name,
    is_feasible_or_better, wall_time_s) by default, or with get_solution=True
    a 4th element: a {course_code: slot} dict if a solution was found, else
    None. Since there is no objective, ANY solution CP-SAT returns is
    equally valid -- there is no "best" to distinguish among feasible
    schedules at a given K, only feasible vs. not.

    enforce_c3=False (default): only C1+C2+C4 are hard -- the pure
        graph-colouring-with-capacity question ("how many colours does
        this conflict graph need at all").
    enforce_c3=True: C1+C2+C3+C4 are ALL hard (C3 forced to exactly zero,
        not merely minimized). Use this to replicate the QUBO/SA pipeline's
        own hybrid-repair acceptance criterion, which only accepts a local
        k_rep once C1=C2=C3=C4=0 is reached -- so the "minimum k" question
        that matters for certifying k_rep is this stricter one, not the
        C3-ignoring chromatic number.
    """
    n = len(courses_df)
    enrollment = courses_df['enrollment'].tolist()
    year = courses_df['year'].tolist()

    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f'x_{i}_{k}') for k in range(K)] for i in range(n)]
    for i in range(n):
        model.AddExactlyOne(x[i])
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adjacency[i, j]]
    for (i, j) in edges:
        for k in range(K):
            model.AddAtMostOne([x[i][k], x[j][k]])
    for k in range(K):
        model.Add(sum(enrollment[i] * x[i][k] for i in range(n)) <= capacity)

    if enforce_c3:
        same_year_edges = [(i, j) for (i, j) in edges if year[i] == year[j]]
        for (i, j) in same_year_edges:
            for k in range(K - 1):
                model.Add(x[i][k] + x[j][k + 1] <= 1)
                model.Add(x[i][k + 1] + x[j][k] <= 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = num_workers

    start = time.time()
    status = solver.Solve(model)
    wall_time = time.time() - start
    status_name = solver.StatusName(status)
    ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    if not get_solution:
        return status_name, ok, round(wall_time, 2)

    solution = None
    if ok:
        solution = {
            courses_df.iloc[i]['course_code']: k
            for i in range(n) for k in range(K) if solver.Value(x[i][k]) == 1
        }
    return status_name, ok, round(wall_time, 2), solution


def find_minimum_k(courses_df, adjacency, capacity, k_min=1, k_max=None,
                    time_limit_per_k_s=60, num_workers=8, verbose=True,
                    enforce_c3=False):
    """Binary search for the smallest number of slots K for which a
    schedule satisfying the hard constraints exists at all (see
    _feasible_at_k for what enforce_c3 changes).

    Feasibility is monotonic in K (a K-slot solution is trivially also a
    valid K+1-slot solution, leaving the extra slot empty), so binary
    search is valid and needs only O(log K) solves.

    Returns a dict with the smallest PROVEN-feasible K found (if any),
    the largest PROVEN-infeasible K found (a certified lower bound even
    if the search doesn't fully close), and the per-K solver log.
    """
    n = len(courses_df)
    if k_max is None:
        k_max = n  # trivially feasible: one exam per slot

    log = []
    lo, hi = k_min, k_max
    proven_infeasible_at = k_min - 1  # certified lower bound: K <= this is impossible
    proven_feasible_at = None         # smallest K certified to work

    # Make sure k_max itself is feasible (sanity bound); if not, the
    # capacity constraint alone makes the instance infeasible regardless
    # of K, which is worth surfacing rather than silently binary-searching
    # forever.
    status, ok, wt = _feasible_at_k(courses_df, adjacency, k_max, capacity,
                                     time_limit_per_k_s, num_workers, enforce_c3)
    log.append({'K': k_max, 'status': status, 'wall_time_s': wt})
    if verbose:
        print(f"  K={k_max:4d} (sanity upper bound): {status} ({wt}s)")
    if status == 'INFEASIBLE':
        return {
            'min_k_found': None,
            'certified_infeasible_up_to': k_max,
            'message': (
                f'Even K={k_max} slots is infeasible -- capacity {capacity} is '
                f'too tight for this instance regardless of slot count. '
                f'Raise --capacity.'
            ),
            'log': log,
        }
    if not ok:
        return {
            'min_k_found': None,
            'certified_infeasible_up_to': proven_infeasible_at,
            'message': (
                f'Could not even establish feasibility at K={k_max} within '
                f'{time_limit_per_k_s}s per K. Increase --time-limit-per-k-s.'
            ),
            'log': log,
        }
    proven_feasible_at = k_max

    while lo < hi:
        mid = (lo + hi) // 2
        status, ok, wt = _feasible_at_k(courses_df, adjacency, mid, capacity,
                                         time_limit_per_k_s, num_workers, enforce_c3)
        log.append({'K': mid, 'status': status, 'wall_time_s': wt})
        if verbose:
            print(f"  K={mid:4d}: {status} ({wt}s)")
        if status == 'FEASIBLE' or status == 'OPTIMAL':
            hi = mid
            proven_feasible_at = mid
        elif status == 'INFEASIBLE':
            lo = mid + 1
            proven_infeasible_at = max(proven_infeasible_at, mid)
        else:
            # UNKNOWN at this K: can't move either bound safely; stop here
            # rather than guess.
            break

    return {
        'min_k_found': proven_feasible_at,
        'certified_infeasible_up_to': proven_infeasible_at,
        'search_closed': proven_feasible_at is not None
                          and proven_infeasible_at == proven_feasible_at - 1,
        'log': log,
    }


def validate_assignment(assignment, courses_df, adjacency, K, capacity):
    """Independently recompute C1-C4 violation counts for a given
    course_code -> slot assignment, so results from any source (CP-SAT,
    the QUBO/SA pipeline's own timetable, etc.) can be checked and
    compared on a common footing.
    """
    code_to_idx = {row['course_code']: i for i, row in courses_df.iterrows()}
    n = len(courses_df)
    enrollment = courses_df['enrollment'].tolist()
    year = courses_df['year'].tolist()

    slot_of = {}
    missing = []
    for code, idx in code_to_idx.items():
        if code in assignment:
            slot_of[idx] = int(assignment[code])
        else:
            missing.append(code)

    c1_violations = len(missing)

    c2_violations = 0
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adjacency[i, j]]
    for (i, j) in edges:
        if i in slot_of and j in slot_of and slot_of[i] == slot_of[j]:
            c2_violations += 1

    c3_violations = 0
    for (i, j) in edges:
        if year[i] == year[j] and i in slot_of and j in slot_of:
            if abs(slot_of[i] - slot_of[j]) == 1:
                c3_violations += 1

    slot_load = {}
    for idx, k in slot_of.items():
        slot_load[k] = slot_load.get(k, 0) + enrollment[idx]
    c4_over_capacity_slots = sum(1 for load in slot_load.values() if load > capacity)
    max_slot_load = max(slot_load.values()) if slot_load else 0

    return {
        'c1_unassigned_exams': c1_violations,
        'c2_conflict_violations': c2_violations,
        'c3_consecutive_violations': c3_violations,
        'c4_slots_over_capacity': c4_over_capacity_slots,
        'max_slot_load': max_slot_load,
        'slots_used': len(slot_load),
    }
