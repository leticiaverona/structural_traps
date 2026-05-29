"""
analysis.py
===========
Paper-specific analytical computations that combine transition matrices and
the household panel to produce the results reported in Sections 4.1–4.3.

Public entry points
-------------------
calculate_quarterly_incidence(df)
    Compute the percentage of households deprived in each indicator,
    per quarter, across the full observation period.

compute_gradient_inputs(P_earlier, P_later, N_earlier, N_later, ...)
    For each indicator and each Hamming-weight stratum, compute the
    change in resolution probability between two periods.

build_indicator_table(periods, ...)
    Aggregate resolution-probability changes into the summary Table 3
    of the paper (trajectory and gradient labels).

state_occupancy_year(df, year, ...)
    Compute the average share of households in each state for a given year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import INDICATORS


# ---------------------------------------------------------------------------
# Internal helpers (not exported)
# ---------------------------------------------------------------------------

def _state_indicators(state_idx: int, n: int = 5) -> tuple:
    """Convert an integer state index to a tuple of binary indicator values.

    Uses MSB-first convention: bit position 0 is d1, position n−1 is d_n.

    Parameters
    ----------
    state_idx : int
        Integer index in [0, 2^n).
    n : int
        Number of indicators.

    Returns
    -------
    tuple of int
        (d1, d2, ..., dn), each 0 or 1.
    """
    return tuple((state_idx >> (n - 1 - i)) & 1 for i in range(n))


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """Compute weighted quantiles via interpolation on the cumulative weight CDF.

    Reduces to numpy.quantile when weights are uniform.

    Parameters
    ----------
    values, weights : 1-D arrays
    quantiles : 1-D array of floats in [0, 1]

    Returns
    -------
    1-D array of quantile values (NaN if input is empty).
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    if values.size == 0:
        return np.full_like(quantiles, np.nan)

    sorter = np.argsort(values)
    values, weights = values[sorter], weights[sorter]
    cumw = np.cumsum(weights)
    if cumw[-1] <= 0:
        return np.full_like(quantiles, np.nan)
    cumw /= cumw[-1]
    return np.interp(quantiles, cumw, values)


# ---------------------------------------------------------------------------
# Quarterly deprivation incidence
# ---------------------------------------------------------------------------

def calculate_quarterly_incidence(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the percentage of deprived households per indicator per quarter.

    Parameters
    ----------
    df : pd.DataFrame
        Household panel with columns 'Ano', 'Trimestre', and the five
        deprivation indicators from config.INDICATORS.

    Returns
    -------
    pd.DataFrame
        One row per (year, quarter) with:
          - one column per indicator (incidence as percentage 0–100)
          - 'Period': pd.Timestamp for plotting (start of quarter)
    """
    grouped = df.groupby(['Ano', 'Trimestre'])[INDICATORS].mean() * 100
    grouped = grouped.reset_index()
    grouped['month'] = (grouped['Trimestre'] - 1) * 3 + 1
    grouped['Period'] = pd.to_datetime(
        grouped[['Ano', 'month']].assign(day=1).rename(columns={'Ano': 'year', 'month': 'month'})
    )
    return grouped.sort_values('Period')


# ---------------------------------------------------------------------------
# Resolution gradient
# ---------------------------------------------------------------------------

def compute_gradient_inputs(
    P_earlier: np.ndarray,
    P_later: np.ndarray,
    N_earlier,
    N_later,
    n_indicators: int = 5,
    min_n_per_state: int = 100,
    quantile_lower: float = 0.25,
    quantile_upper: float = 0.75,
) -> tuple[dict, dict]:
    """Compute per-indicator resolution-probability changes by Hamming-weight stratum.

    For indicator d_k, the resolution probability from origin state s is:
        P_resolve(d_k | s) = Σ_{s': d_k(s')=0} P[s, s']

    The change (later − earlier) is computed per origin state, then
    aggregated over Hamming-weight strata h = 1 … n_indicators using
    N-weighted averages and weighted quantile bands.

    Parameters
    ----------
    P_earlier, P_later : ndarray (2^n, 2^n)
        Row-stochastic transition matrices for the two periods.
    N_earlier, N_later : array-like (2^n,) or (2^n, 2^n)
        Origin-state observation counts.  Matrices are summed over
        destinations to yield per-origin totals.
    n_indicators : int
        Number of binary indicators; state space = 2^n_indicators.
    min_n_per_state : int
        Origin states with fewer observations in either period are excluded.
    quantile_lower, quantile_upper : float
        Band quantiles for the dispersion ribbon (default: IQR).

    Returns
    -------
    deltas : dict {'d1': [v_h1, ..., v_hn], ...}
        Weighted-mean Δ P_resolve per Hamming-weight stratum, in pp.
    bands : dict {'d1': {'lower': [...], 'upper': [...]}, ...}
        Weighted quantile band endpoints, in pp.
    """
    P_earlier = np.asarray(P_earlier, dtype=float)
    P_later = np.asarray(P_later, dtype=float)
    N_earlier = np.asarray(N_earlier, dtype=float)
    N_later = np.asarray(N_later, dtype=float)

    if N_earlier.ndim == 2:
        N_earlier = N_earlier.sum(axis=1)
    if N_later.ndim == 2:
        N_later = N_later.sum(axis=1)

    n_states = 1 << n_indicators
    if P_earlier.shape != (n_states, n_states) or P_later.shape != (n_states, n_states):
        raise ValueError(f"Expected ({n_states}, {n_states}) matrices.")
    if N_earlier.shape != (n_states,) or N_later.shape != (n_states,):
        raise ValueError(f"Expected ({n_states},) count vectors.")

    ind_per_state = np.array([_state_indicators(s, n_indicators) for s in range(n_states)], dtype=int)
    h_per_state = ind_per_state.sum(axis=1)
    N_combined = N_earlier + N_later
    valid_state = (N_earlier >= min_n_per_state) & (N_later >= min_n_per_state)

    deltas: dict = {}
    bands: dict = {}

    for k in range(n_indicators):
        key = f'd{k + 1}'
        origin_deprived = ind_per_state[:, k] == 1
        resolve_mask = (ind_per_state[:, k] == 0).astype(float)

        Pres_earlier = P_earlier @ resolve_mask
        Pres_later = P_later @ resolve_mask
        delta_per_state = (Pres_later - Pres_earlier) * 100.0  # percentage points

        eligible = origin_deprived & valid_state
        delta_h, lower_h, upper_h = [], [], []

        for h in range(1, n_indicators + 1):
            mask = eligible & (h_per_state == h)
            if not mask.any() or N_combined[mask].sum() <= 0:
                delta_h.append(np.nan)
                lower_h.append(np.nan)
                upper_h.append(np.nan)
                continue

            values = delta_per_state[mask]
            weights = N_combined[mask]
            delta_h.append(float(np.average(values, weights=weights)))

            if values.size > 1:
                ql, qu = _weighted_quantile(values, weights, np.array([quantile_lower, quantile_upper]))
                lower_h.append(float(ql))
                upper_h.append(float(qu))
            else:
                lower_h.append(np.nan)
                upper_h.append(np.nan)

        deltas[key] = delta_h
        bands[key] = {'lower': lower_h, 'upper': upper_h}

    return deltas, bands


# ---------------------------------------------------------------------------
# Indicator transition summary table (Table 3)
# ---------------------------------------------------------------------------

def build_indicator_table(
    periods: list,
    indicator_labels: dict | list | None = None,
    n_indicators: int = 5,
    min_n_per_state: int = 100,
    weighted: bool = True,
    resolution: str = 'clean',
    units: str = 'prob',
    stagnation_thr_pp: float = 1.0,
    gradient_slope_thr_pp: float = 0.5,
    gradient_report: str = 'last',
) -> tuple[pd.DataFrame, dict]:
    """Build the indicator-level resolution-probability change table (Table 3).

    Parameters
    ----------
    periods : list of (label, P, N)
        Ordered sequence of (period label, P matrix, count matrix/vector).
        Length ≥ 2; consecutive pairs define the comparison periods.
    indicator_labels : dict or list or None
        Optional mapping {'d1': 'Literacy', ...} or list of names.
    n_indicators : int
        Number of binary indicators.
    min_n_per_state : int
        Minimum count per state per period to include in the average.
    weighted : bool
        If True, use N-weighted averages (recommended).
    resolution : str
        'aggregate' — resolution probability sums over all destinations
                      with the indicator off (any exit route counts).
        'clean'     — resolution probability is the single cell P[s, s'],
                      where s' matches s except bit k is 0 (only direct
                      resolution, holding all other deprivations constant).
    units : str
        'prob' — output as probability change (e.g. 0.056).
        'pp'   — output as percentage points (e.g. 5.6).
    stagnation_thr_pp : float
        |Avg Δ| (pp) below this threshold in ALL pairs → 'Stagnation'.
    gradient_slope_thr_pp : float
        |slope| (pp per deprivation count) below this → gradient is 'flat'.
    gradient_report : str
        'last' — Gradient column shows the label for the most recent pair.
        'all'  — shows labels for every pair separated by ' / '.

    Returns
    -------
    table : pd.DataFrame
        Summary with columns Indicator, Avg Δ per pair, Trajectory, Gradient.
    detail : dict
        Raw averages (pp) and gradient labels per pair, for full diagnostics.
    """
    if len(periods) < 2:
        raise ValueError("Need ≥2 periods to form at least one comparison.")
    if resolution not in ('aggregate', 'clean'):
        raise ValueError("resolution must be 'aggregate' or 'clean'.")

    n_states = 1 << n_indicators
    ind = np.array([_state_indicators(s, n_indicators) for s in range(n_states)], dtype=int)
    h_per_state = ind.sum(axis=1)

    def _vec_n(N):
        N = np.asarray(N, dtype=float)
        if N.ndim == 2:
            N = N.sum(axis=1)
        if N.shape != (n_states,):
            raise ValueError(f"N has shape {N.shape}; expected ({n_states},)")
        return N

    P_seq = [(lab, np.asarray(P, dtype=float), _vec_n(N)) for lab, P, N in periods]
    keys = [f'd{k + 1}' for k in range(n_indicators)]

    pair_labels, avg_pp, perh_pp, slope_pp, glabel = [], {k: [] for k in keys}, {k: [] for k in keys}, {k: [] for k in keys}, {k: [] for k in keys}

    for (la, Pa, Na), (lb, Pb, Nb) in zip(P_seq[:-1], P_seq[1:]):
        pair_labels.append(f'{la}\u2192{lb}')
        valid = (Na >= min_n_per_state) & (Nb >= min_n_per_state)
        Ncomb = Na + Nb

        for k in range(n_indicators):
            key = keys[k]
            if resolution == 'aggregate':
                resolve = (ind[:, k] == 0).astype(float)
                ra, rb = Pa @ resolve, Pb @ resolve
            else:
                src = np.arange(n_states)
                tgt = src & ~(1 << (n_indicators - 1 - k))
                ra, rb = Pa[src, tgt], Pb[src, tgt]

            dps = (rb - ra) * 100.0
            elig = (ind[:, k] == 1) & valid

            if elig.any() and Ncomb[elig].sum() > 0:
                avg = float(np.average(dps[elig], weights=Ncomb[elig])) if weighted else float(np.mean(dps[elig]))
            else:
                avg = np.nan
            avg_pp[key].append(avg)

            vh = []
            for h in range(1, n_indicators + 1):
                m = elig & (h_per_state == h)
                vh.append(
                    float(np.average(dps[m], weights=Ncomb[m]))
                    if (m.any() and Ncomb[m].sum() > 0) else np.nan
                )
            perh_pp[key].append(vh)

            pts = [(h, v) for h, v in enumerate(vh, 1) if not np.isnan(v)]
            slope = float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]) if len(pts) >= 2 else np.nan
            slope_pp[key].append(slope)

            if np.isnan(avg) or abs(avg) < stagnation_thr_pp:
                glabel[key].append('inexistent')
            elif np.isnan(slope):
                glabel[key].append('n/a')
            elif slope < -gradient_slope_thr_pp:
                glabel[key].append('decreasing')
            elif slope > gradient_slope_thr_pp:
                glabel[key].append('increasing')
            else:
                glabel[key].append('flat')

    def _name(k_idx, key):
        if isinstance(indicator_labels, dict):
            return indicator_labels.get(key, key)
        if indicator_labels:
            return indicator_labels[k_idx]
        return key

    conv = (lambda x: round(x / 100.0, 4)) if units == 'prob' else (lambda x: round(x, 2))
    rows = []
    for k in range(n_indicators):
        key = keys[k]
        avgs = avg_pp[key]
        finite = [a for a in avgs if not np.isnan(a)]
        if not finite:
            traj = 'n/a'
        elif all(abs(a) < stagnation_thr_pp for a in avgs):
            traj = 'Stagnation'
        elif abs(avgs[-1]) < stagnation_thr_pp and any(a > stagnation_thr_pp for a in finite):
            traj = 'Stalled'
        elif all(a > 0 for a in finite):
            traj = 'Accelerating' if (len(avgs) >= 2 and avgs[-1] > avgs[0]) else 'Decelerating'
        elif all(a < 0 for a in finite):
            traj = 'Declining'
        else:
            traj = 'Mixed'

        gcol = glabel[key][-1] if gradient_report == 'last' else ' / '.join(glabel[key])
        row = {'Indicator': _name(k, key)}
        for plab, a in zip(pair_labels, avgs):
            row[f'Avg \u0394 {plab}'] = (np.nan if np.isnan(a) else conv(a))
        row['Trajectory'] = traj
        row['Gradient'] = gcol
        rows.append(row)

    table = pd.DataFrame(rows)
    detail = {
        'pair_labels': pair_labels,
        'avg_delta_pp': avg_pp,
        'per_h_delta_pp': perh_pp,
        'gradient_slope_pp': slope_pp,
        'gradient_label_per_pair': glabel,
        'units': units, 'weighted': weighted, 'resolution': resolution,
        'thresholds': {
            'stagnation_thr_pp': stagnation_thr_pp,
            'gradient_slope_thr_pp': gradient_slope_thr_pp,
        },
    }
    return table, detail


# ---------------------------------------------------------------------------
# State occupancy
# ---------------------------------------------------------------------------

def state_occupancy_year(
    df: pd.DataFrame,
    year: int = 2025,
    state_col: str = 'deprivation_profile',
    year_col: str = 'Ano',
    quarter_col: str = 'Trimestre',
    id_col: str = 'id_dom',
    n_states: int = 32,
) -> pd.DataFrame:
    """Compute the average household share per state for a given year.

    For each quarter in the specified year, counts unique households per
    state, computes the quarter share, then averages across quarters.

    Parameters
    ----------
    df : pd.DataFrame
    year : int
    n_states : int
        Expected number of states (default 32 = 2^5).

    Returns
    -------
    pd.DataFrame
        One row per state, sorted by avg_count descending, with columns:
        binary (profile string), avg_count, avg_share, min_count, max_count.
    """
    d = df[df[year_col] == year].copy()
    if d.empty:
        raise ValueError(f"No records for {year_col}=={year}.")

    # Normalise state to integer 0…n_states−1
    s = d[state_col]
    if s.dtype == object:
        d[state_col] = s.astype(str).str.zfill(5).map(lambda b: int(b, 2))
    else:
        d[state_col] = s.round().astype(int)

    if not d[state_col].between(0, n_states - 1).any():
        raise ValueError(
            f"No '{state_col}' values fall in [0, {n_states - 1}]. "
            f"Check state encoding (examples: {d[state_col].unique()[:5]})."
        )

    quarters = sorted(d[quarter_col].unique())
    idx = pd.RangeIndex(n_states, name=state_col)
    counts_df = pd.DataFrame(index=idx)
    shares_df = pd.DataFrame(index=idx)

    for q in quarters:
        dq = d[d[quarter_col] == q]
        if id_col is not None:
            dq = dq.drop_duplicates(subset=[id_col, quarter_col])
        c = dq[state_col].value_counts().reindex(idx, fill_value=0).astype(int)
        counts_df[q] = c
        total = c.sum()
        shares_df[q] = c / total if total else 0.0

    out = pd.DataFrame(index=idx)
    out['binary'] = [format(s, '05b') for s in idx]
    out['avg_count'] = counts_df.mean(axis=1)
    out['avg_share'] = shares_df.mean(axis=1)
    out['min_count'] = counts_df.min(axis=1)
    out['max_count'] = counts_df.max(axis=1)

    return out.sort_values('avg_count', ascending=False)
