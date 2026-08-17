"""
Dataset construction for the optimality-checking tool.

Filtering and conflict-graph construction here intentionally mirror
`multi_state_qubo_V2/QAOA/run_exam_scheduler.py`
(`generate_dataset_from_csv` / `_build_courses_and_adjacency_from_rows`)
so that a CP-SAT run on the "all" graph with the same --k / --capacity
is checking the *same* instance the QUBO+SA+repair pipeline solved,
not a look-alike one. If the filtering rules in that file change,
update this module to match.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    'Registration No.', 'Course Code', 'Academic Session', 'Registration Status',
    'Course Classification', 'Course Type', 'Semester', 'Description',
]

ACADEMIC_SESSIONS = ['JUL-NOV 2025', 'WINTER 2025']
EXCLUDED_SEMESTERS = ['XI', 'XII']

SEMESTER_TO_NUMERIC = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
}


def load_and_filter_csv(input_csv, max_rows=None):
    """Load the enrollment CSV and apply the same filters as the QUBO pipeline:
    Approved + Theory + {JUL-NOV 2025, WINTER 2025} sessions, excluding
    semesters XI/XII.
    """
    src = pd.read_csv(input_csv)
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("--max-rows must be a positive integer")
        src = src.head(max_rows).copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in src.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    for col in REQUIRED_COLUMNS:
        src[col] = src[col].astype(str).str.strip()

    base = src[
        (src['Registration Status'].str.upper() == 'APPROVED')
        & (src['Course Classification'].str.upper() == 'THEORY')
        & (src['Academic Session'].isin(ACADEMIC_SESSIONS))
    ].copy()
    base = base[~base['Semester'].isin(EXCLUDED_SEMESTERS)].copy()

    if base.empty:
        raise ValueError("No rows left after filters (Approved + Theory + JUL-NOV/WINTER, excl. XI/XII).")

    return base


def build_courses_and_adjacency(filtered_df, adjacency_mode='all'):
    """Build the courses table and conflict adjacency matrix.

    adjacency_mode:
        'major' -> only Course Type == MAJOR rows
        'all'   -> all filtered theory rows (MAJOR + ELECTIVE + ...)

    Conflict rule: two courses conflict if at least one student is enrolled
    in both (same as the QUBO pipeline's C2 conflict graph).
    """
    if adjacency_mode == 'major':
        rows = filtered_df[filtered_df['Course Type'].str.upper() == 'MAJOR'].copy()
        if rows.empty:
            raise ValueError("No MAJOR rows left after filtering.")
    elif adjacency_mode == 'all':
        rows = filtered_df.copy()
    else:
        raise ValueError("adjacency_mode must be 'major' or 'all'")

    course_col = 'Course Code'
    student_col = 'Registration No.'

    course_codes = sorted(rows[course_col].astype(str).str.strip().unique().tolist())
    course_to_id = {code: idx for idx, code in enumerate(course_codes)}

    sem_numeric = pd.to_numeric(
        rows['Semester'].astype(str).str.strip().map(SEMESTER_TO_NUMERIC), errors='coerce'
    )
    tmp = rows.copy()
    tmp['semester_num'] = sem_numeric

    enroll_count = tmp.groupby(course_col)[student_col].nunique().to_dict()
    semester_mode = tmp.groupby(course_col)['semester_num'].agg(
        lambda s: int(s.mode().iloc[0]) if s.notna().any() and not s.mode().empty else 1
    ).to_dict()

    courses = []
    for code in course_codes:
        courses.append({
            'course_id': course_to_id[code],
            'course_code': code,
            'year': int(semester_mode.get(code, 1)),
            'enrollment': int(enroll_count.get(code, 0)),
        })
    courses_df = pd.DataFrame(courses).sort_values('course_id').reset_index(drop=True)

    n = len(course_codes)
    adjacency = np.zeros((n, n), dtype=np.int64)

    grouped = tmp.groupby(student_col)[course_col].apply(lambda s: sorted(set(s.tolist())))
    for _student_id, courses_for_student in grouped.items():
        for c1, c2 in itertools.combinations(courses_for_student, 2):
            i, j = course_to_id[c1], course_to_id[c2]
            adjacency[i, j] = 1
            adjacency[j, i] = 1

    return courses_df, adjacency


def select_subset(courses_df, adjacency, num_exams=None, sample_mode='random', seed=42):
    """Select an induced subgraph of `num_exams` courses from the full instance.

    sample_mode:
        'random'  -> uniform random sample of courses (seeded, reproducible)
        'densest' -> the num_exams courses with the highest conflict degree
                     (a harder, denser stress-test instance)
        'first'   -> the first num_exams courses by course_code order

    If num_exams is None or >= number of available courses, the full
    instance is returned unchanged.
    """
    n = len(courses_df)
    if num_exams is None or num_exams >= n:
        return courses_df.reset_index(drop=True), adjacency

    if num_exams <= 0:
        raise ValueError("--num-exams must be a positive integer")

    if sample_mode == 'random':
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=num_exams, replace=False))
    elif sample_mode == 'densest':
        degrees = adjacency.sum(axis=1)
        idx = np.sort(np.argsort(-degrees)[:num_exams])
    elif sample_mode == 'first':
        idx = np.arange(num_exams)
    else:
        raise ValueError("sample_mode must be 'random', 'densest', or 'first'")

    sub_courses = courses_df.iloc[idx].reset_index(drop=True)
    sub_adjacency = adjacency[np.ix_(idx, idx)]
    return sub_courses, sub_adjacency


def select_by_course_codes(courses_df, adjacency, course_codes):
    """Select the exact induced subgraph for a given list of course codes.

    Use this (instead of select_subset's random/densest/first sampling)
    to certify optimality/minimality on a *specific, real* subset -- e.g.
    the exact repair set the QUBO/SA pipeline's hybrid repair procedure
    identified in an actual run, so the certificate is about the literal
    instance the paper reports, not a look-alike sample of the same size.
    """
    code_to_row = {row['course_code']: i for i, row in courses_df.iterrows()}
    missing = [c for c in course_codes if c not in code_to_row]
    if missing:
        raise ValueError(f"Course codes not found in this instance: {missing}")
    idx = [code_to_row[c] for c in course_codes]
    sub_courses = courses_df.iloc[idx].reset_index(drop=True)
    sub_adjacency = adjacency[np.ix_(idx, idx)]
    return sub_courses, sub_adjacency


def load_saved_run_instance(run_dir):
    """Load courses.csv + conflict_adjacency.csv already saved by a prior
    QUBO/SA pipeline run (output/run_.../{all,major}/), so a subset check
    can be built from the *exact* instance that run solved, without
    re-deriving it from the raw enrollment CSV.
    """
    run_dir = Path(run_dir)
    courses_df = pd.read_csv(run_dir / 'courses.csv')
    adjacency = pd.read_csv(run_dir / 'conflict_adjacency.csv').values.astype(np.int64)
    return courses_df, adjacency


def graph_summary(courses_df, adjacency):
    n = len(courses_df)
    num_edges = int(adjacency.sum() // 2)
    density = (num_edges / (n * (n - 1) / 2) * 100) if n > 1 else 0.0
    degrees = adjacency.sum(axis=1)
    max_degree = int(degrees.max()) if n > 0 else 0
    return {
        'num_courses': n,
        'num_conflict_edges': num_edges,
        'density_pct': round(density, 3),
        'max_degree': max_degree,
        # max_degree + 1 is a GREEDY UPPER bound on the chromatic number
        # (Brooks' theorem), NOT a lower bound. A valid K-colouring can
        # exist for K well below this; it is only a rough sizing hint.
        'greedy_upper_bound_k': max_degree + 1,
    }
