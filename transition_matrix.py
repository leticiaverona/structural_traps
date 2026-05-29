"""
transition_matrix.py
====================
Functions for building Markov transition matrices from the household
deprivation panel, including valid-pair extraction and helper utilities.

The state space consists of all 2^5 = 32 binary vectors d1 d2 d3 d4 d5,
encoded as five-character strings ('00000' … '11111', MSB-first).  The
bit ordering follows INDICATORS in config.py.

A transition X_t → X_{t+1} is valid if:
    (1) same household id_dom (ensured by groupby shift within id);
    (2) consecutive quarters: period_{t+1} − period_t = 1
        (with period = year × 4 + (quarter − 1)).

Public entry points
-------------------
generate_all_binary_states(n_bits=5)
    Return all 2^n_bits profile strings in lexicographic order.

build_transition_matrix(df, ...)
    Estimate the row-stochastic transition matrix P and the count matrix N
    from a household panel.  Returns (P, counts, meta) where meta contains
    diagnostics.

sort_matrix_by_severity(P_df)
    Reorder P rows and columns by Hamming weight (number of deprivations),
    then alphabetically within each weight class.

build_transitions(df, ...)
    Extract all valid one-quarter transition pairs from the panel, enforcing
    both temporal and interview-number contiguity.  Returns a DataFrame of
    (id_dom, state_t, next_state) rows used by bootstrap_convergence and
    test_markov_property.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product


# ---------------------------------------------------------------------------
# State-space utilities
# ---------------------------------------------------------------------------

def generate_all_binary_states(n_bits: int = 5) -> list[str]:
    """Return all 2^n_bits binary profile strings, zero-padded to n_bits digits.

    Example with n_bits=3: ['000', '001', '010', '011', '100', '101', '110', '111']
    """
    return [''.join(bits) for bits in product('01', repeat=n_bits)]


def sort_matrix_by_severity(P_df: pd.DataFrame) -> pd.DataFrame:
    """Sort a transition matrix by Hamming weight (deprivation count), then lexicographically.

    Parameters
    ----------
    P_df : pd.DataFrame
        Square transition matrix whose index and columns are binary profile strings.

    Returns
    -------
    pd.DataFrame
        P_df reindexed so that states appear in order:
        all states with 0 deprivations, then 1, …, then n deprivations,
        with alphabetical order as tie-breaker within each group.
    """
    profiles = list(P_df.index)
    sorted_profiles = sorted(profiles, key=lambda x: (x.count('1'), x))
    return P_df.loc[sorted_profiles, sorted_profiles]


# ---------------------------------------------------------------------------
# Main transition matrix builder
# ---------------------------------------------------------------------------

def build_transition_matrix(
    df: pd.DataFrame,
    id_col: str = 'id_dom',
    state_col: str = 'deprivation_profile',
    year_col: str = 'year',
    quarter_col: str = 'quarter',
    n_bits: int = 5,
    start_year: int | None = None,
    start_quarter: int | None = None,
    end_year: int | None = None,
    end_quarter: int | None = None,
    min_interviews: int = 2,
    enforce_one_step: bool = True,
    zero_row_policy: str = 'absorbing',   # 'absorbing' | 'keep'
    epsilon: float | None = None,
    normalize: bool = True,
    row_thresholds: tuple = (30, 100, 200, 500),
    cell_thresholds: tuple = (30, 100),
    show_smallest: int = 10,
    verbose: bool = True,
):
    """Estimate a first-order Markov transition matrix from the household panel.

    A transition X_t → X_{t+1} is included if:
        - Both observations belong to the same household (id_col);
        - Consecutive quarters: period_{t+1} − period_t = 1
          (where period = year × 4 + quarter − 1), if enforce_one_step=True;
        - Both period_t and period_{t+1} lie within [start, end] if a window
          is specified.

    Parameters
    ----------
    df : pd.DataFrame
        Household panel in long format (one row per quarter observation).
    id_col : str
        Column identifying the household across quarters.
    state_col : str
        Column containing the binary profile string (e.g. '01011').
    year_col, quarter_col : str
        Columns for year and quarter (both integer).
    n_bits : int
        Number of indicators; defines the state space as 2^n_bits profiles.
    start_year, start_quarter, end_year, end_quarter : int or None
        Optional window restricting which transitions are counted.
        Both endpoints of each transition must fall inside the window.
    min_interviews : int
        Households with fewer total appearances in df are excluded before
        transition counting.
    enforce_one_step : bool
        If True (recommended), discard transitions with period gaps > 1.
    zero_row_policy : str
        How to handle states with no observed outgoing transitions:
            'absorbing' — set the diagonal to 1 (self-absorbing state);
            'keep'      — leave the row as zeros.
    epsilon : float or None
        If provided, adds this value to all counts before normalisation
        (Laplace smoothing).  None (default) uses pure MLE.
    normalize : bool
        If True, return row-normalised probabilities; if False, raw counts.
    row_thresholds, cell_thresholds : tuple of int
        Count thresholds for the diagnostic summary.
    show_smallest : int
        Number of lowest-count origin states to list in verbose output.
    verbose : bool
        Print diagnostic summary if True.

    Returns
    -------
    P : pd.DataFrame
        S×S row-stochastic transition probability matrix (or count matrix
        if normalize=False), indexed by binary profile strings.
    counts : pd.DataFrame
        S×S raw count matrix N_ij.
    meta : dict
        Diagnostic metadata: family counts, transition counts, zero-row
        states, row-sum statistics, stochasticity check, etc.
    """
    if zero_row_policy not in ('absorbing', 'keep'):
        raise ValueError("zero_row_policy must be 'absorbing' or 'keep'.")

    S = 2 ** n_bits
    all_states = generate_all_binary_states(n_bits)
    state_index = pd.Index(all_states, name=state_col)

    # --- 1. Validate and copy ---
    for c in (id_col, state_col, year_col, quarter_col):
        if c not in df.columns:
            raise ValueError(f"Column not found: '{c}'")

    df = df.copy()
    df[state_col] = df[state_col].astype(str)
    df[year_col] = df[year_col].astype(int)
    df[quarter_col] = df[quarter_col].astype(int)

    # Verify all observed states fit in the declared state space
    unknown = set(df[state_col].unique()) - set(all_states)
    if unknown:
        raise ValueError(
            f"{len(unknown)} observed state(s) outside the {S}-state space "
            f"(n_bits={n_bits}): {sorted(unknown)[:5]}. "
            f"Check zero-padding of '{state_col}'."
        )

    # --- 2. Continuous period index ---
    # period = year * 4 + (quarter - 1); consecutive quarters differ by 1.
    df['_period'] = df[year_col] * 4 + (df[quarter_col] - 1)

    start_code = (
        (start_year * 4 + (start_quarter - 1)) if start_year is not None
        else int(df['_period'].min())
    )
    end_code = (
        (end_year * 4 + (end_quarter - 1)) if end_year is not None
        else int(df['_period'].max())
    )

    # --- 3. Panel-level minimum interviews filter ---
    n_int = df.groupby(id_col).size()
    valid_ids = n_int[n_int >= min_interviews].index
    df_f = df[df[id_col].isin(valid_ids)].sort_values([id_col, '_period']).copy()
    n_families_panel = df_f[id_col].nunique()

    # --- 4. Build (X_t, X_{t+1}) pairs via shift within household groups ---
    g = df_f.groupby(id_col, sort=False)
    df_f['_next_state'] = g[state_col].shift(-1)
    df_f['_next_period'] = g['_period'].shift(-1)

    raw_mask = df_f['_next_state'].notna()
    n_raw_pairs = int(raw_mask.sum())

    # --- 5. Contiguity check ---
    if enforce_one_step:
        one_step = (df_f['_next_period'] - df_f['_period']) == 1
    else:
        one_step = raw_mask.copy()
    n_gap_discarded = int((raw_mask & ~one_step).sum())

    # --- 6. Window filter: both endpoints must lie in [start_code, end_code] ---
    origin_in = (df_f['_period'] >= start_code) & (df_f['_period'] <= end_code)
    dest_in = (df_f['_next_period'] >= start_code) & (df_f['_next_period'] <= end_code)
    n_boundary_crossers = int((raw_mask & one_step & (origin_in ^ dest_in)).sum())

    valid_mask = raw_mask & one_step & origin_in & dest_in
    pairs = (
        df_f.loc[valid_mask, [state_col, '_next_state']]
        .rename(columns={'_next_state': 'next_state'})
    )
    n_transitions = len(pairs)
    n_families_window = df_f.loc[valid_mask, id_col].nunique()

    # --- 7. Count matrix: full fixed S×S space ---
    if n_transitions:
        counts = pd.crosstab(pairs[state_col], pairs['next_state'])
    else:
        counts = pd.DataFrame(
            index=pd.Index([], name=state_col),
            columns=pd.Index([], name='next_state'),
        )
    counts = counts.reindex(index=state_index, columns=state_index, fill_value=0).astype(int)

    # --- 8. Optional Laplace smoothing ---
    counts_for_norm = (counts + float(epsilon)) if epsilon else counts

    # --- 9. Row-normalise + zero-row policy ---
    row_sums = counts.sum(axis=1)
    zero_row_states = list(row_sums[row_sums == 0].index)
    n_zero_rows = len(zero_row_states)

    if normalize:
        denom = counts_for_norm.sum(axis=1)
        P = counts_for_norm.div(denom.replace(0, np.nan), axis=0)
        if not epsilon and zero_row_policy == 'absorbing':
            for s in zero_row_states:
                P.loc[s] = 0.0
                P.loc[s, s] = 1.0
        P = P.fillna(0.0)
    else:
        P = counts.copy()

    # --- 10. Diagnostic summaries ---
    row_n_summary: dict = {
        'min':    int(row_sums.min()),
        'p10':    float(row_sums.quantile(0.10)),
        'p25':    float(row_sums.quantile(0.25)),
        'median': float(row_sums.median()),
        'mean':   float(row_sums.mean()),
        'max':    int(row_sums.max()),
    }
    for thr in row_thresholds:
        row_n_summary[f'n_rows_below_{thr}'] = int((row_sums < thr).sum())
    smallest_origins = row_sums.sort_values().head(show_smallest)

    cell_n = counts.to_numpy().ravel()
    cell_n_summary: dict = {
        'total_cells': int(cell_n.size),
        'non_zero':    int((cell_n > 0).sum()),
    }
    for thr in cell_thresholds:
        cell_n_summary[f'cells_above_{thr}'] = int((cell_n >= thr).sum())
        cell_n_summary[f'cells_above_{thr}_pct'] = float(100 * (cell_n >= thr).mean())

    is_stochastic = bool(np.allclose(P.sum(axis=1).to_numpy(), 1.0)) if normalize else None

    meta = {
        'n_families_panel':    n_families_panel,
        'n_families_window':   n_families_window,
        'n_obs':               len(df_f),
        'n_raw_pairs':         n_raw_pairs,
        'n_gap_discarded':     n_gap_discarded,
        'n_boundary_crossers': n_boundary_crossers,
        'n_transitions':       n_transitions,
        'n_states':            S,
        'states':              all_states,
        'zero_rows':           n_zero_rows,
        'zero_row_states':     zero_row_states,
        'zero_row_policy':     (f'epsilon={epsilon}' if epsilon else zero_row_policy),
        'epsilon':             epsilon,
        'enforce_one_step':    enforce_one_step,
        'is_stochastic':       is_stochastic,
        'row_sums':            row_sums,
        'row_n_summary':       row_n_summary,
        'cell_n_summary':      cell_n_summary,
        'smallest_origins':    smallest_origins,
        'min_interviews':      min_interviews,
        'start_code':          start_code,
        'end_code':            end_code,
    }

    # --- 11. Verbose summary ---
    if verbose:
        denom_raw = max(n_raw_pairs, 1)
        print(f"Window (period code) : {start_code} → {end_code}")
        print(f"Households (panel)   : {n_families_panel:>12,}")
        print(f"Households (window)  : {n_families_window:>12,}")
        print(f"Observations         : {len(df_f):>12,}")
        print(f"Raw pairs            : {n_raw_pairs:>12,}")
        print(
            f"  discarded (gap)    : {n_gap_discarded:>12,}  "
            f"({100 * n_gap_discarded / denom_raw:.2f}%)"
            + ("  <- contiguity enforced" if enforce_one_step else "  (check disabled)")
        )
        print(
            f"  boundary crossers  : {n_boundary_crossers:>12,}  "
            f"({100 * n_boundary_crossers / denom_raw:.2f}%)"
        )
        print(
            f"Valid transitions    : {n_transitions:>12,}  "
            f"({100 * n_transitions / denom_raw:.2f}%)"
        )
        print(f"States (fixed space) : {S:>12}")
        print(f"Zero-row states      : {n_zero_rows:>12}   policy={meta['zero_row_policy']}")
        print(f"Is stochastic        : {str(is_stochastic):>12}")
        print()
        print("Row-origin N summary:")
        for k in ('min', 'p10', 'p25', 'median', 'mean', 'max'):
            print(f"  {k:<8}         : {row_n_summary[k]:>12,.0f}")
        print()
        print("Origin states below each N threshold:")
        for thr in row_thresholds:
            print(f"  N < {thr:>4}          : {row_n_summary[f'n_rows_below_{thr}']:>3} / {S}")
        print()
        print(f"Top {show_smallest} origin states with fewest observations:")
        for state, n in smallest_origins.items():
            print(f"  {str(state):<8}          : {int(n):>12,}")
        print()
        print(f"Cell counts ({S * S} = {S}×{S}):")
        print(f"  non-zero           : {cell_n_summary['non_zero']:>12,} / {cell_n_summary['total_cells']:,}")
        for thr in cell_thresholds:
            print(
                f"  N ≥ {thr:>4}         : {cell_n_summary[f'cells_above_{thr}']:>12,}"
                f"  ({cell_n_summary[f'cells_above_{thr}_pct']:.1f}%)"
            )

    return P, counts, meta


# ---------------------------------------------------------------------------
# Valid transition-pair extractor (used by bootstrap and Markov tests)
# ---------------------------------------------------------------------------

def build_transitions(
    df: pd.DataFrame,
    id_col: str = 'id_dom',
    year_col: str = 'Ano',
    quarter_col: str = 'Trimestre',
    state_col: str = 'deprivation_profile',
    interview_col: str = 'interview_number',
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Extract all valid one-quarter transition pairs from the household panel.

    A pair (t, t+1) is valid if:
        - The next observation belongs to the same household;
        - period_{t+1} − period_t = 1  (consecutive quarters);
        - interview_number_{t+1} − interview_number_t = 1  (if the column exists).

    Both checks guard against rotation-panel attrition and cross-household
    contamination from the 'last observation carries forward' logic.

    Parameters
    ----------
    df : pd.DataFrame
        Household panel in long format.
    id_col, year_col, quarter_col, state_col, interview_col : str
        Column names.
    verbose : bool
        Print diagnostic counts.

    Returns
    -------
    df_trans : pd.DataFrame
        Valid transition rows with columns [id_col, year_col, quarter_col,
        interview_col, state_col, 'next_state'].
    diag : dict
        Diagnostic counts: raw pairs, valid pairs, discarded by each check.
    """
    df = df.sort_values([id_col, year_col, quarter_col]).copy()
    df['_period'] = df[year_col].astype(int) * 4 + (df[quarter_col].astype(int) - 1)

    g = df.groupby(id_col, sort=False)
    df['_next_state'] = g[state_col].shift(-1)
    df['_next_period'] = g['_period'].shift(-1)

    has_interview = interview_col in df.columns
    if has_interview:
        df['_next_interview'] = g[interview_col].shift(-1)

    raw_mask = df['_next_state'].notna()
    n_raw_pairs = int(raw_mask.sum())

    period_ok = (df['_next_period'] - df['_period']) == 1
    if has_interview:
        interview_ok = (df['_next_interview'] - df[interview_col]) == 1
        valid_mask = raw_mask & period_ok & interview_ok
    else:
        valid_mask = raw_mask & period_ok

    df_trans = df.loc[valid_mask].copy()
    df_trans = df_trans.rename(columns={'_next_state': 'next_state'})

    diag = {
        'raw_pairs_after_shift': n_raw_pairs,
        'valid_transitions':     int(len(df_trans)),
        'discarded_period_gap':  int((raw_mask & ~period_ok).sum()),
        'discarded_interview_gap': (
            int((raw_mask & period_ok & ~interview_ok).sum())
            if has_interview else None
        ),
        'fraction_kept':         (len(df_trans) / n_raw_pairs) if n_raw_pairs else 0.0,
        'used_interview_check':  has_interview,
    }

    if verbose:
        print("build_transitions:")
        print(f"  Raw (id, id+1) pairs       : {diag['raw_pairs_after_shift']:,}")
        print(f"  Valid (contiguous) pairs   : {diag['valid_transitions']:,}")
        print(f"  Discarded by period gap    : {diag['discarded_period_gap']:,}")
        if has_interview:
            print(f"  Discarded by interview gap : {diag['discarded_interview_gap']:,}")
        print(f"  Fraction kept              : {diag['fraction_kept'] * 100:.2f}%")

    # Drop auxiliary columns
    drop_cols = [c for c in ['_period', '_next_period', '_next_interview'] if c in df_trans.columns]
    df_trans = df_trans.drop(columns=drop_cols)

    return df_trans, diag


# ---------------------------------------------------------------------------
# Low-level count helper (used by bootstrap and Markov order tests)
# ---------------------------------------------------------------------------

def _compute_P_counts(
    from_idx: np.ndarray,
    to_idx: np.ndarray,
    n_states: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a transition matrix from integer index arrays.

    Parameters
    ----------
    from_idx, to_idx : 1-D integer arrays
        Origin and destination state indices for each observed transition.
    n_states : int
        Total number of states (defines the matrix dimension).

    Returns
    -------
    P : ndarray (n_states, n_states)
        Row-stochastic transition probability matrix.
    counts : ndarray (n_states, n_states)
        Raw transition count matrix.
    row_sums : ndarray (n_states,)
        Number of observed outgoing transitions per origin state.
    """
    counts = np.zeros((n_states, n_states), dtype=np.int64)
    np.add.at(counts, (from_idx, to_idx), 1)
    row_sums = counts.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(row_sums[:, None] > 0, counts / row_sums[:, None], 0.0)
    return P, counts, row_sums
