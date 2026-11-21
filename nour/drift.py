"""Population Stability Index and data drift reporting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(reference, current, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Calculate Population Stability Index for continuous numeric distributions.

    Interior edges are quantiles of the reference distribution.
    Floors bin proportions at `eps` and computes sum((c - r) * ln(c / r)).
    """
    ref_arr = np.asarray(reference, dtype=float).ravel()
    cur_arr = np.asarray(current, dtype=float).ravel()

    ref = ref_arr[~np.isnan(ref_arr)]
    cur = cur_arr[~np.isnan(cur_arr)]

    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("reference and current must not be empty after dropping NaNs")

    q_points = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(ref, q_points))

    k = len(edges) + 1
    ref_bins = np.searchsorted(edges, ref, side="right")
    cur_bins = np.searchsorted(edges, cur, side="right")

    r = np.bincount(ref_bins, minlength=k).astype(float) / len(ref)
    c = np.bincount(cur_bins, minlength=k).astype(float) / len(cur)

    r = np.maximum(r, eps)
    c = np.maximum(c, eps)

    val = np.sum((c - r) * np.log(c / r))
    return float(val)


def categorical_psi(reference, current, eps: float = 1e-4) -> float:
    """Calculate Population Stability Index for categorical distributions.

    Categories are the union of unique values across both sets.
    """
    if isinstance(reference, pd.Series):
        ref_series = reference.dropna()
    else:
        ref_series = pd.Series(reference).dropna()

    if isinstance(current, pd.Series):
        cur_series = current.dropna()
    else:
        cur_series = pd.Series(current).dropna()

    ref = ref_series.astype(str).tolist()
    cur = cur_series.astype(str).tolist()

    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("reference and current must not be empty after dropping missing values")

    categories = sorted(set(ref).union(set(cur)))

    from collections import Counter
    ref_counts = Counter(ref)
    cur_counts = Counter(cur)

    r = np.array([ref_counts[cat] for cat in categories], dtype=float) / len(ref)
    c = np.array([cur_counts[cat] for cat in categories], dtype=float) / len(cur)

    r = np.maximum(r, eps)
    c = np.maximum(c, eps)

    val = np.sum((c - r) * np.log(c / r))
    return float(val)


def drift_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    threshold: float = 0.2,
) -> dict:
    """Generate drift report across specified numeric and categorical columns."""
    columns = {}

    for col in numeric:
        m_ref = float(ref_df[col].isna().mean())
        m_cur = float(cur_df[col].isna().mean())
        col_psi = psi(ref_df[col], cur_df[col])
        drifted = bool(col_psi >= threshold or abs(m_cur - m_ref) >= 0.1)
        columns[col] = {
            "psi": col_psi,
            "missing_rate_ref": m_ref,
            "missing_rate_cur": m_cur,
            "drifted": drifted,
        }

    for col in categorical:
        m_ref = float(ref_df[col].isna().mean())
        m_cur = float(cur_df[col].isna().mean())
        col_psi = categorical_psi(ref_df[col], cur_df[col])
        drifted = bool(col_psi >= threshold or abs(m_cur - m_ref) >= 0.1)
        columns[col] = {
            "psi": col_psi,
            "missing_rate_ref": m_ref,
            "missing_rate_cur": m_cur,
            "drifted": drifted,
        }

    drifted_columns = sorted([col for col, d in columns.items() if d["drifted"]])

    return {
        "columns": columns,
        "drifted_columns": drifted_columns,
    }
