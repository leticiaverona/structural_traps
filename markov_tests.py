"""
markov_tests.py
===============
Statistical validation functions for the Markov chain analysis.

Sections
--------
1. Temporal homogeneity (LR test across periods)
2. Pairwise effect sizes between period matrices
3. Bootstrap convergence diagnostic
4. Markov-order tests (Anderson-Goodman, Chapman-Kolmogorov)
5. Predictive validation (in-sample multi-horizon + hold-out)

All functions operate on the outputs of build_transition_matrix() and
build_transitions() from transition_matrix.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
from scipy.stats import chi2

from transition_matrix import build_transitions, _compute_P_counts


# ============================================================================
# 1. Temporal homogeneity — likelihood ratio test
# ============================================================================

def lr_test_homogeneity(counts_dict: dict) -> dict:
    """Likelihood-ratio test for temporal homogeneity of a Markov chain.

    Tests H0: P_m = P_pool for all periods m (time-homogeneous chain)
    against H1: at least one period matrix differs from the pooled estimate.
    Follows Bickenbach & Bode (2003), generalising Anderson-Goodman (1957).

    Test statistic:
        Λ = 2 · Σ_m Σ_i Σ_{j: N_m(i,j)>0}  N_m(i,j) · log(P_m(i,j) / P_pool(i,j))

    Under H0, Λ ~ χ²(df) with:
        df = Σ_i (M − 1) · (K_i − 1)
    where K_i = number of destinations with N_pool(i,j) > 0 (structural zeros
    are excluded from the degrees of freedom).

    Parameters
    ----------
    counts_dict : dict {period_name: pd.DataFrame of shape S×S}
        Raw (un-normalised) count matrices per period.  All matrices must
        share the same index and column labels (same state ordering).

    Returns
    -------
    dict with keys:
        lr          : test statistic Λ
        df          : degrees of freedom
        p_value     : p-value under χ²(df)
        lr_per_period : contribution of each period to Λ
        n_total     : total transitions in the pooled matrix
        n_per_period : transitions per period
        m_periods   : number of periods M
        k_per_row   : number of non-zero destinations per origin (pooled)
        P_pool      : pooled transition probability matrix
        counts_pool : pooled count matrix
    """
    period_names = list(counts_dict.keys())
    M = len(period_names)
    if M < 2:
        raise ValueError("Need at least 2 periods for the homogeneity test.")

    states = counts_dict[period_names[0]].index.tolist()
    counts_clean = {}
    for name, counts in counts_dict.items():
        if list(counts.index) != states or list(counts.columns) != states:
            raise ValueError(f"Matrix '{name}' has different index/columns from others.")
        counts_clean[name] = counts.fillna(0)

    counts_pool = sum(counts_clean.values())
    row_sums_pool = counts_pool.sum(axis=1)
    P_pool = counts_pool.div(row_sums_pool.replace(0, np.nan), axis=0).fillna(0)
    n_total = int(counts_pool.values.sum())

    P_pool_arr = P_pool.values
    lr_per_period: dict = {}
    n_per_period: dict = {}
    lr_total = 0.0

    for name, counts in counts_clean.items():
        counts_arr = counts.values.astype(float)
        row_sums_m = counts_arr.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            P_m_arr = np.where(row_sums_m > 0, counts_arr / row_sums_m, 0.0)

        mask = counts_arr > 0
        ratio = np.where(mask, P_m_arr / np.where(P_pool_arr > 0, P_pool_arr, 1.0), 1.0)
        contrib = np.where(mask, counts_arr * np.log(ratio), 0.0)

        lr_m = 2.0 * contrib.sum()
        lr_per_period[name] = lr_m
        n_per_period[name] = int(counts_arr.sum())
        lr_total += lr_m

    k_per_row = (counts_pool > 0).sum(axis=1)
    df = int(((M - 1) * (k_per_row - 1).clip(lower=0)).sum())
    p_value = float(chi2.sf(lr_total, df))

    print("=" * 60)
    print(f"LR homogeneity test — {M} periods")
    print("=" * 60)
    print(f"Periods          : {period_names}")
    for name in period_names:
        print(f"  N({name:<10}) = {n_per_period[name]:>10,}   contrib Λ = {lr_per_period[name]:>12,.2f}")
    print(f"N (pooled)       : {n_total:>10,}")
    print(f"Λ                : {lr_total:>12,.2f}")
    print(f"df               : {df:>12,}")
    print(f"p-value          : {p_value:.4g}")
    print(f"K_i avg (destinations): {k_per_row.mean():.1f}   min={k_per_row.min()}   max={k_per_row.max()}")
    print("=" * 60)

    return {
        'lr': lr_total, 'df': df, 'p_value': p_value,
        'lr_per_period': lr_per_period, 'n_total': n_total,
        'n_per_period': n_per_period, 'm_periods': M,
        'k_per_row': k_per_row, 'P_pool': P_pool, 'counts_pool': counts_pool,
    }


# ============================================================================
# 2. Pairwise effect sizes between period matrices
# ============================================================================

def effect_size_pairwise(
    matrices_dict: dict,
    min_count: int = 30,
    period_pairs: list | None = None,
    delta_thresholds: tuple = (0.01, 0.05, 0.10),
) -> dict:
    """Pairwise effect-size analysis between period transition matrices.

    For each pair of periods (m, m'), computes:
        |Δ|(i,j) = |P_m(i,j) − P_m'(i,j)|             (cell-level)
        TVD_i = (1/2) Σ_j |P_m(i,j) − P_m'(i,j)|      (row-level)

    Cell-level statistics are restricted to cells where both periods have
    N ≥ min_count; TVD is computed over all 32 columns.

    Parameters
    ----------
    matrices_dict : dict {name: {'P': DataFrame, 'counts': DataFrame}}
        Output of build_transition_matrix, one entry per period.
    min_count : int
        Minimum count in BOTH periods for a cell to enter the cell summary.
    period_pairs : list of (name_a, name_b) or None
        Pairs to compare.  If None, all combinations are evaluated.
    delta_thresholds : tuple of float
        |Δ| thresholds for count reporting.

    Returns
    -------
    dict {(name_a, name_b): result_dict}
        Each result contains delta_matrix, cell_summary, tvd_per_row, etc.
    """
    names = list(matrices_dict.keys())
    if period_pairs is None:
        period_pairs = list(combinations(names, 2))

    results: dict = {}

    for name_a, name_b in period_pairs:
        P_a = matrices_dict[name_a]['P']
        P_b = matrices_dict[name_b]['P']
        N_a = matrices_dict[name_a]['counts'].fillna(0)
        N_b = matrices_dict[name_b]['counts'].fillna(0)

        delta = (P_a - P_b).abs()
        mask = (N_a >= min_count) & (N_b >= min_count)
        cells_compared = int(mask.values.sum())
        cells_total = int(mask.size)

        delta_filtered = delta.where(mask).stack().dropna()

        if len(delta_filtered) > 0:
            cell_summary = {
                'mean':   float(delta_filtered.mean()),
                'median': float(delta_filtered.median()),
                'p75':    float(delta_filtered.quantile(0.75)),
                'p90':    float(delta_filtered.quantile(0.90)),
                'p99':    float(delta_filtered.quantile(0.99)),
                'max':    float(delta_filtered.max()),
            }
            for thr in delta_thresholds:
                cell_summary[f'n_above_{thr}'] = int((delta_filtered > thr).sum())
                cell_summary[f'pct_above_{thr}'] = float(100 * (delta_filtered > thr).mean())
        else:
            cell_summary = {k: float('nan') for k in ['mean', 'median', 'p75', 'p90', 'p99', 'max']}

        tvd_per_row = 0.5 * delta.sum(axis=1)
        tvd_mean = float(tvd_per_row.mean())
        tvd_max = float(tvd_per_row.max())
        max_state_key = tvd_per_row.idxmax()
        tvd_max_state = str(max_state_key)

        row_counts = pd.DataFrame({
            f'N_{name_a}': N_a.sum(axis=1),
            f'N_{name_b}': N_b.sum(axis=1),
        })
        n_a_max = int(row_counts.loc[max_state_key, f'N_{name_a}'])
        n_b_max = int(row_counts.loc[max_state_key, f'N_{name_b}'])

        results[(name_a, name_b)] = {
            'delta_matrix':   delta,
            'mask_min_count': mask,
            'cells_compared': cells_compared,
            'cells_total':    cells_total,
            'cell_summary':   cell_summary,
            'tvd_per_row':    tvd_per_row,
            'tvd_mean':       tvd_mean,
            'tvd_max':        tvd_max,
            'tvd_max_state':  tvd_max_state,
            'n_a_max_state':  n_a_max,
            'n_b_max_state':  n_b_max,
            'row_counts':     row_counts,
        }

        print(f"\n{'=' * 64}")
        print(f"Effect size:  {name_a}  vs  {name_b}")
        print(f"{'=' * 64}")
        print(f"Cells compared       : {cells_compared:>5} / {cells_total} ({100 * cells_compared / cells_total:.1f}%)")
        print()
        print(f"Distribution of |Δ| (cells with N ≥ {min_count} in both periods):")
        for k in ('mean', 'median', 'p75', 'p90', 'p99', 'max'):
            print(f"  {k:<8}         : {cell_summary[k]:.4f}")
        print()
        print("Cells above each |Δ| threshold:")
        for thr in delta_thresholds:
            print(f"  |Δ| > {thr:>5.2f}     : {cell_summary[f'n_above_{thr}']:>4}  "
                  f"({cell_summary[f'pct_above_{thr}']:.2f}% of compared cells)")
        print()
        print("TVD per row-origin (all 32 columns):")
        print(f"  mean             : {tvd_mean:.4f}")
        print(f"  max              : {tvd_max:.4f}   "
              f"(state: {tvd_max_state}, N_{name_a}={n_a_max:,}, N_{name_b}={n_b_max:,})")
        print(f"  n with TVD > 0.05: {int((tvd_per_row > 0.05).sum())}")
        print(f"  n with TVD > 0.10: {int((tvd_per_row > 0.10).sum())}")

    return results


# ============================================================================
# 3. Bootstrap convergence diagnostic
# ============================================================================

def bootstrap_convergence(
    df: pd.DataFrame,
    sample_sizes: list[int] | None = None,
    n_replicates: int = 100,
    min_n_origin: int = 100,
    epsilon: float = 0.02,
    id_col: str = 'id_dom',
    year_col: str = 'Ano',
    quarter_col: str = 'Trimestre',
    state_col: str = 'deprivation_profile',
    interview_col: str = 'interview_number',
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Convergence diagnostic for the transition matrix as a function of sample size.

    For each N in sample_sizes, draws B subsamples of N households without
    replacement, re-estimates P, and measures the median row-wise TVD against
    the full-sample P.  Reports the smallest N where the sustained convergence
    criterion (all larger N also below epsilon) is met.

    Parameters
    ----------
    df : pd.DataFrame
        Household panel (one row per quarter observation).
    sample_sizes : list of int or None
        Household counts to evaluate.  If None, uses a log-spaced grid from
        1% to 80% of total households.
    n_replicates : int
        Bootstrap replicates per sample size.
    min_n_origin : int
        Origin states with fewer than this many counts in the full matrix
        are excluded from the TVD summary.
    epsilon : float
        Convergence threshold on median row-wise TVD.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        curve              : DataFrame with median, IQR, 5/95 pct TVD per sample size.
        records            : DataFrame, one row per (size, replicate).
        P_full             : full-sample transition matrix.
        states             : ordered state labels.
        row_sums_full      : origin counts in the full matrix.
        converged_at       : smallest N meeting the sustained criterion (or None).
        safety_ratio       : n_families_total / converged_at.
        epsilon            : threshold used.
        n_families_total   : total unique households in valid transitions.
        n_transitions_total: total valid transitions.
        diag_transitions   : diagnostics from build_transitions.
    """
    rng = np.random.default_rng(seed)

    df_trans, diag_trans = build_transitions(
        df, id_col=id_col, year_col=year_col, quarter_col=quarter_col,
        state_col=state_col, interview_col=interview_col, verbose=verbose,
    )

    states = sorted(set(df_trans[state_col]).union(set(df_trans['next_state'])))
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    from_idx = df_trans[state_col].map(state_to_idx).to_numpy()
    to_idx = df_trans['next_state'].map(state_to_idx).to_numpy()
    ids = df_trans[id_col].to_numpy()

    P_full, _, row_sums_full = _compute_P_counts(from_idx, to_idx, n_states)
    rows_keep = row_sums_full >= min_n_origin

    unique_ids, inverse = np.unique(ids, return_inverse=True)
    n_families = len(unique_ids)

    order = np.argsort(inverse, kind='stable')
    sorted_inv = inverse[order]
    boundaries = np.concatenate(([0], np.flatnonzero(np.diff(sorted_inv)) + 1, [len(order)]))

    if sample_sizes is None:
        lo = max(100, int(0.01 * n_families))
        hi = max(lo + 1, int(0.8 * n_families))
        sample_sizes = sorted(set(
            int(round(x)) for x in np.logspace(np.log10(lo), np.log10(hi), 10)
        ))
    sample_sizes = [s for s in sample_sizes if 1 <= s <= n_families]

    if verbose:
        print()
        print("bootstrap_convergence:")
        print(f"  Unique households  : {n_families:,}")
        print(f"  Valid transitions  : {len(df_trans):,}")
        print(f"  States             : {n_states}")
        print(f"  Rows with N ≥ {min_n_origin}: {int(rows_keep.sum())}")
        print(f"  Replicates per N   : {n_replicates}")
        print(f"  Sample sizes       : {sample_sizes}")
        print()

    records = []
    for size in sample_sizes:
        for b in range(n_replicates):
            picked = rng.choice(n_families, size=size, replace=False)
            pos_chunks = [order[boundaries[k]:boundaries[k + 1]] for k in picked]
            positions = np.concatenate(pos_chunks)

            P_sample, _, row_sums_sample = _compute_P_counts(
                from_idx[positions], to_idx[positions], n_states
            )
            tvd_row = 0.5 * np.abs(P_full - P_sample).sum(axis=1)
            valid = rows_keep & (row_sums_sample > 0)
            if valid.sum() == 0:
                continue

            records.append({
                'n_families': int(size), 'n_transitions': int(len(positions)),
                'replicate': int(b), 'median_tvd': float(np.median(tvd_row[valid])),
                'mean_tvd': float(np.mean(tvd_row[valid])),
                'n_valid_rows': int(valid.sum()),
            })

        if verbose:
            sub = [r['median_tvd'] for r in records if r['n_families'] == size]
            if sub:
                med = float(np.median(sub))
                q25, q75 = float(np.percentile(sub, 25)), float(np.percentile(sub, 75))
                print(f"  N = {size:>7,} households  ->  median TVD = {med:.4f}  [IQR {q25:.4f}–{q75:.4f}]")

    df_records = pd.DataFrame(records)
    curve = (
        df_records.groupby('n_families').agg(
            median=('median_tvd', 'median'),
            q05=('median_tvd', lambda x: float(np.percentile(x, 5))),
            q25=('median_tvd', lambda x: float(np.percentile(x, 25))),
            q75=('median_tvd', lambda x: float(np.percentile(x, 75))),
            q95=('median_tvd', lambda x: float(np.percentile(x, 95))),
            mean_n_transitions=('n_transitions', 'mean'),
        ).reset_index().sort_values('n_families')
    )

    converged_at = None
    for _, row in curve.iterrows():
        n = int(row['n_families'])
        tail = curve.loc[curve['n_families'] >= n, 'median']
        if (tail < epsilon).all():
            converged_at = n
            break

    safety_ratio = (n_families / converged_at) if converged_at else None

    if verbose:
        print()
        print(f"  Criterion: median TVD < {epsilon} (sustained for all larger N)")
        if converged_at is not None:
            print(f"  Converged at N = {converged_at:,} households")
            print(f"  Safety ratio   = {safety_ratio:.2f}x  (N_total / N_converged)")
        else:
            print(f"  Did not converge within the tested range.")

    return {
        'curve': curve, 'records': df_records, 'P_full': P_full,
        'states': states, 'row_sums_full': row_sums_full,
        'converged_at': converged_at, 'safety_ratio': safety_ratio,
        'epsilon': epsilon, 'min_n_origin': min_n_origin,
        'n_families_total': n_families, 'n_transitions_total': len(df_trans),
        'diag_transitions': diag_trans,
    }


# ============================================================================
# 4. Markov-order tests
# ============================================================================

def extract_subsequences(
    df: pd.DataFrame,
    length: int = 3,
    id_col: str = 'id_dom',
    year_col: str = 'Ano',
    quarter_col: str = 'Trimestre',
    state_col: str = 'deprivation_profile',
    interview_col: str = 'interview_number',
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Extract all contiguous subsequences of given length from the panel.

    A subsequence of length L starting at time t is valid if all L−1
    consecutive steps satisfy both the period-contiguity and
    interview-number-increment conditions.

    Parameters
    ----------
    length : int
        Subsequence length (≥2).  Length 3 is used for Anderson-Goodman.

    Returns
    -------
    sequences : ndarray (n_subseq, length) int32
        State indices for each valid subsequence.
    states : ndarray
        Ordered state labels.
    state_to_idx : dict
    diag : dict
        Extraction diagnostics.
    """
    if length < 2:
        raise ValueError("length must be at least 2")

    df = df.sort_values([id_col, year_col, quarter_col]).copy()
    df['_period'] = df[year_col].astype(int) * 4 + (df[quarter_col].astype(int) - 1)

    states = np.array(sorted(df[state_col].unique()))
    state_to_idx = {s: i for i, s in enumerate(states)}
    df['_state_idx'] = df[state_col].map(state_to_idx).astype(np.int32)

    has_interview = interview_col in df.columns
    periods_arr = df['_period'].to_numpy()
    state_idx_arr = df['_state_idx'].to_numpy()
    interview_arr = df[interview_col].to_numpy() if has_interview else None

    indices_per_id = df.groupby(id_col, sort=False).indices

    seq_list = []
    n_contributing = n_too_short = n_only_gaps = 0

    for _, idx in indices_per_id.items():
        T = len(idx)
        if T < length:
            n_too_short += 1
            continue

        pers = periods_arr[idx]
        step_valid = np.diff(pers) == 1
        if has_interview:
            intv = interview_arr[idx]
            step_valid &= np.diff(intv) == 1

        n_starts = T - length + 1
        starts_valid = step_valid[:n_starts].copy()
        for k in range(1, length - 1):
            starts_valid &= step_valid[k:n_starts + k]

        valid_t = np.flatnonzero(starts_valid)
        if len(valid_t) == 0:
            n_only_gaps += 1
            continue

        sts = state_idx_arr[idx]
        for t in valid_t:
            seq_list.append(sts[t:t + length])
        n_contributing += 1

    sequences = np.stack(seq_list).astype(np.int32) if seq_list else np.zeros((0, length), dtype=np.int32)

    diag = {
        'n_subsequences': int(len(sequences)),
        'n_families_total': len(indices_per_id),
        'n_families_contributing': n_contributing,
        'n_families_too_short': n_too_short,
        'n_families_only_gaps': n_only_gaps,
        'length': length, 'n_states': len(states),
        'used_interview_check': has_interview,
    }

    if verbose:
        avg = diag['n_subsequences'] / diag['n_families_contributing'] if n_contributing else 0.0
        print("extract_subsequences:")
        print(f"  Households (total)                : {diag['n_families_total']:,}")
        print(f"  Households contributing           : {n_contributing:,}")
        print(f"  Households too short (<{length} obs) : {n_too_short:,}")
        print(f"  Households with only gapped seqs  : {n_only_gaps:,}")
        print(f"  Subsequences of length {length}          : {diag['n_subsequences']:,}")
        print(f"  Avg subsequences per HH           : {avg:.2f}")
        print(f"  State space size                  : {diag['n_states']}")

    return sequences, states, state_to_idx, diag


def count_transitions(sequences: np.ndarray, S: int) -> tuple[np.ndarray, np.ndarray]:
    """Count first- and second-order transitions from a subsequence array.

    Parameters
    ----------
    sequences : ndarray (n_subseq, T)
        Integer state-index arrays.
    S : int
        Number of states.

    Returns
    -------
    n1 : ndarray (S, S)     — n1[j, k] = # times j→k
    n2 : ndarray (S, S, S)  — n2[i, j, k] = # times i,j→k
    """
    _, T = sequences.shape
    n1 = np.zeros((S, S), dtype=np.int64)
    for t in range(T - 1):
        np.add.at(n1, (sequences[:, t], sequences[:, t + 1]), 1)

    n2 = np.zeros((S, S, S), dtype=np.int64)
    for t in range(T - 2):
        np.add.at(n2, (sequences[:, t], sequences[:, t + 1], sequences[:, t + 2]), 1)

    return n1, n2


def count_twostep(sequences: np.ndarray, S: int) -> np.ndarray:
    """Count two-step transitions t → t+2 for the Chapman-Kolmogorov test."""
    _, T = sequences.shape
    n_2step = np.zeros((S, S), dtype=np.int64)
    for t in range(T - 2):
        np.add.at(n_2step, (sequences[:, t], sequences[:, t + 2]), 1)
    return n_2step


def anderson_goodman_test(n1: np.ndarray, n2: np.ndarray, S: int) -> dict:
    """Anderson-Goodman (1957) likelihood-ratio test for the first-order Markov property.

    H0: chain is first-order Markov (P(X_{t+2}|X_{t+1}) = P(X_{t+2}|X_{t+1}, X_t))

    Returns
    -------
    dict with lr_stat, df_theoretical, df_effective, p_theoretical, p_effective.
    """
    n_j_dot = n1.sum(axis=1)
    n_ij_dot = n2.sum(axis=2)

    N2  = n2
    NJ  = np.broadcast_to(n_j_dot[np.newaxis, :, np.newaxis], (S, S, S)).copy()
    NIJ = np.broadcast_to(n_ij_dot[:, :, np.newaxis],         (S, S, S)).copy()
    NJK = np.broadcast_to(n1[np.newaxis, :, :],               (S, S, S)).copy()

    mask = (N2 > 0) & (NJ > 0) & (NIJ > 0) & (NJK > 0)
    log_ratio = np.zeros((S, S, S))
    log_ratio[mask] = np.log(N2[mask] * NJ[mask] / (NIJ[mask] * NJK[mask]))

    lr_stat = 2.0 * np.sum(N2 * log_ratio)
    df_theoretical = S * (S - 1) ** 2
    df_effective = int((n_ij_dot > 0).sum()) * (S - 1)

    return {
        'lr_stat':       lr_stat,
        'df_theoretical': df_theoretical,
        'df_effective':   df_effective,
        'p_theoretical': float(1 - stats.chi2.cdf(lr_stat, df_theoretical)),
        'p_effective':   float(1 - stats.chi2.cdf(lr_stat, df_effective)),
    }


def chapman_kolmogorov_test(n1: np.ndarray, n_2step: np.ndarray, S: int) -> dict:
    """Compare the empirical 2-step matrix with P² predicted by first-order chain.

    Returns MAE and RMSE between the two probability matrices.  Smaller values
    are more consistent with the first-order Markov hypothesis.
    """
    row_sum1 = n1.sum(axis=1, keepdims=True)
    P1 = np.where(row_sum1 > 0, n1 / row_sum1, 0)

    row_sum2 = n_2step.sum(axis=1, keepdims=True)
    P2_emp = np.where(row_sum2 > 0, n_2step / row_sum2, 0)
    P2_pred = P1 @ P1

    diff = P2_emp - P2_pred
    return {
        'P1': P1, 'P2_emp': P2_emp, 'P2_pred': P2_pred,
        'MAE': float(np.mean(np.abs(diff))),
        'RMSE': float(np.sqrt(np.mean(diff ** 2))),
    }


def effect_size(n1: np.ndarray, n2: np.ndarray, S: int, min_obs: int = 30) -> dict:
    """Weighted MAE between first- and second-order transition probabilities.

    Parameters
    ----------
    min_obs : int
        Minimum pair (i,j) count to include in the 'dense' summary.

    Returns
    -------
    dict with mae_all, mae_dense, max_diff_all, max_diff_dense,
    n_pairs_total, n_pairs_dense.
    """
    row1 = n1.sum(axis=1, keepdims=True)
    P1 = n1 / np.where(row1 > 0, row1, 1)

    n_ij = n2.sum(axis=2)
    row2 = n_ij[:, :, np.newaxis]
    P2 = n2 / np.where(row2 > 0, row2, 1)
    P2[n_ij == 0] = 0

    dense_mask = n_ij >= min_obs
    diff = np.abs(P2 - P1[np.newaxis, :, :])

    total_obs = n2.sum()
    w_all = n2 / total_obs if total_obs > 0 else np.zeros_like(n2, dtype=float)
    mae_all = float(np.sum(w_all * diff))

    n2_dense = n2 * dense_mask[:, :, np.newaxis]
    total_dense = n2_dense.sum()
    w_dense = n2_dense / total_dense if total_dense > 0 else np.zeros_like(n2, dtype=float)
    mae_dense = float(np.sum(w_dense * diff))

    return {
        'mae_all':         mae_all,
        'mae_dense':       mae_dense,
        'max_diff_all':    float(diff[n2 > 0].max()),
        'max_diff_dense':  float(diff[n2_dense > 0].max()),
        'n_pairs_total':   int((n_ij > 0).sum()),
        'n_pairs_dense':   int(dense_mask.sum()),
        'min_obs':         min_obs,
    }


def test_markov_property(
    df: pd.DataFrame,
    length: int = 3,
    min_obs: int = 30,
    id_col: str = 'id_dom',
    year_col: str = 'Ano',
    quarter_col: str = 'Trimestre',
    state_col: str = 'deprivation_profile',
    interview_col: str = 'interview_number',
) -> dict:
    """Orchestrate the full Markov-order test suite.

    Runs extract_subsequences, then Anderson-Goodman, Chapman-Kolmogorov,
    and effect-size tests, printing a summary of results.

    Returns
    -------
    dict with anderson_goodman, chapman_kolmogorov, effect_size, transition_matrix,
    states, n1, n2, sequences, diag_extraction.
    """
    sequences, states, _, diag = extract_subsequences(
        df, length=length, id_col=id_col, year_col=year_col,
        quarter_col=quarter_col, state_col=state_col,
        interview_col=interview_col, verbose=True,
    )

    S = len(states)
    n1, n2 = count_transitions(sequences, S)
    n_2step = count_twostep(sequences, S)
    ag = anderson_goodman_test(n1, n2, S)
    ck = chapman_kolmogorov_test(n1, n_2step, S)
    es = effect_size(n1, n2, S, min_obs=min_obs)

    print("\n── Anderson-Goodman ───────────────────────────────────────")
    print(f"  LR              = {ag['lr_stat']:,.2f}")
    print(f"  df theoretical  = {ag['df_theoretical']:,}  |  p = {ag['p_theoretical']:.2e}")
    print(f"  df effective    = {ag['df_effective']:,}  |  p = {ag['p_effective']:.2e}")

    print("\n── Chapman-Kolmogorov ─────────────────────────────────────")
    print(f"  MAE  (P² vs P_emp) = {ck['MAE']:.6f}")
    print(f"  RMSE (P² vs P_emp) = {ck['RMSE']:.6f}")

    print("\n── Effect size ────────────────────────────────────────────")
    print(f"  Pairs (i,j) with data          : {es['n_pairs_total']}")
    print(f"  Dense pairs (≥{min_obs} obs)      : {es['n_pairs_dense']}")
    print(f"  Weighted MAE — all             : {es['mae_all']:.6f}")
    print(f"  Weighted MAE — dense           : {es['mae_dense']:.6f}  <- use this")
    print(f"  Max diff — all                 : {es['max_diff_all']:.4f}")
    print(f"  Max diff — dense               : {es['max_diff_dense']:.4f}")

    return {
        'anderson_goodman': ag, 'chapman_kolmogorov': ck, 'effect_size': es,
        'transition_matrix': ck['P1'], 'states': states,
        'n1': n1, 'n2': n2, 'sequences': sequences, 'diag_extraction': diag,
    }


def violation_by_pair(
    n1: np.ndarray, n2: np.ndarray, S: int, states: np.ndarray, min_obs: int = 30
) -> pd.DataFrame:
    """Compute per-pair (i,j) MAE between second- and first-order probabilities.

    Returns a DataFrame sorted by max_diff descending, restricted to pairs
    with n2[i,j,:].sum() >= min_obs.
    """
    row1 = n1.sum(axis=1, keepdims=True)
    P1 = n1 / np.where(row1 > 0, row1, 1)

    n_ij = n2.sum(axis=2)
    row2 = n_ij[:, :, np.newaxis]
    P2 = n2 / np.where(row2 > 0, row2, 1)

    records = []
    for i in range(S):
        for j in range(S):
            if n_ij[i, j] < min_obs:
                continue
            diff_ij = np.abs(P2[i, j, :] - P1[j, :])
            k_max = int(np.argmax(diff_ij))
            records.append({
                'state_i':   states[i], 'state_j': states[j],
                'obs_ij':    int(n_ij[i, j]),
                'mae':       round(float(diff_ij.mean()), 4),
                'max_diff':  round(float(diff_ij.max()), 4),
                'state_k_max': states[k_max],
                'P2_k':      round(float(P2[i, j, k_max]), 4),
                'P1_k':      round(float(P1[j, k_max]), 4),
            })

    return (
        pd.DataFrame(records)
        .sort_values('max_diff', ascending=False)
        .reset_index(drop=True)
    )


def summarize_violations(df_viol: pd.DataFrame) -> None:
    """Print a distributional summary of per-pair Markov violations."""
    mae = df_viol['mae']
    print(f"Pairs analysed   : {len(df_viol)}")
    print(f"MAE median       : {mae.median():.4f}")
    print(f"MAE mean         : {mae.mean():.4f}")
    print(f"MAE p75          : {mae.quantile(.75):.4f}")
    print(f"MAE p90          : {mae.quantile(.90):.4f}")
    print(f"MAE p99          : {mae.quantile(.99):.4f}")
    print("\nMAE distribution by pair:")
    bins = [0, .02, .05, .10, .20, .50, 1.0]
    labels = ['[0, .02)', '[.02,.05)', '[.05,.10)', '[.10,.20)', '[.20,.50)', '[.50,1.0]']
    df_viol = df_viol.copy()
    df_viol['bin'] = pd.cut(mae, bins=bins, labels=labels)
    for label, count in df_viol['bin'].value_counts().sort_index().items():
        print(f"  {label} : {count:4d} {'█' * (count // 3)}")


# ============================================================================
# 5. Predictive validation
# ============================================================================

def _build_pair_transitions(
    df: pd.DataFrame,
    pair: tuple,
    id_col: str,
    year_col: str,
    quarter_col: str,
    state_col: str,
    interview_col: str,
    target_year: int,
) -> pd.DataFrame:
    """Return valid (state_t, state_tp1) rows for one (q_from, q_to) pair."""
    q_from, q_to = pair
    if q_to != q_from + 1:
        raise ValueError(f"Non-consecutive pair: ({q_from}, {q_to})")

    def _slice(q, col_alias):
        cols = [id_col, state_col] + ([interview_col] if interview_col in df.columns else [])
        return (
            df[(df[year_col] == target_year) & (df[quarter_col] == q)][cols]
            .rename(columns={state_col: col_alias})
        )

    a = _slice(q_from, 'state_t')
    b = _slice(q_to, 'state_tp1')
    merged = a.merge(b, on=id_col, how='inner', suffixes=('_t', '_tp1'))
    if interview_col in df.columns:
        merged = merged[merged[f'{interview_col}_tp1'] - merged[f'{interview_col}_t'] == 1]
    return merged[[id_col, 'state_t', 'state_tp1']]


def _matrix_from_transitions(df_trans: pd.DataFrame, states: list) -> tuple:
    """Build P and row counts from a transitions DataFrame."""
    state_to_idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    counts = np.zeros((n, n), dtype=np.int64)
    fi = df_trans['state_t'].map(state_to_idx).to_numpy()
    ti = df_trans['state_tp1'].map(state_to_idx).to_numpy()
    np.add.at(counts, (fi, ti), 1)
    row_sums = counts.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(row_sums[:, None] > 0, counts / row_sums[:, None], 0.0)
    return P, row_sums


def _empirical_marginal(
    df: pd.DataFrame, target_year: int, quarter: int,
    states: list, year_col: str, quarter_col: str, state_col: str,
) -> np.ndarray:
    """Compute the empirical share of households in each state for one quarter."""
    sub = df[(df[year_col] == target_year) & (df[quarter_col] == quarter)]
    counts = sub[state_col].value_counts()
    pi = np.array([counts.get(s, 0) for s in states], dtype=float)
    total = pi.sum()
    return pi / total if total > 0 else pi


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def _mae(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.mean(np.abs(p - q)))


def _correlation(p: np.ndarray, q: np.ndarray) -> float:
    if p.std() == 0 or q.std() == 0:
        return float('nan')
    return float(np.corrcoef(p, q)[0, 1])


def _metrics(pred: np.ndarray, emp: np.ndarray) -> dict:
    return {
        'tvd': _tvd(pred, emp), 'mae': _mae(pred, emp),
        'max_abs_diff': float(np.max(np.abs(pred - emp))),
        'corr': _correlation(pred, emp),
        'pred_mass': float(pred.sum()), 'emp_mass': float(emp.sum()),
    }


def predictive_validation(
    df: pd.DataFrame,
    target_year: int = 2025,
    initial_quarter: int = 1,
    horizons: list = (1, 2, 3),
    holdout_train_pairs: list = ((1, 2), (2, 3)),
    holdout_test_pair: tuple = (3, 4),
    year_col: str = 'Ano',
    quarter_col: str = 'Trimestre',
    state_col: str = 'deprivation_profile',
    id_col: str = 'id_dom',
    interview_col: str = 'interview_number',
    renormalize_after_step: bool = True,
    verbose: bool = True,
) -> dict:
    """Predictive validation of the transition matrix estimated for a given year.

    Two complementary tests:
        A) In-sample multi-horizon: propagate via π · P^k, compare with
           empirical marginals k quarters ahead.
        B) Hold-out: estimate P_train from a subset of transitions, predict
           the held-out quarter's empirical marginal.

    Returns
    -------
    dict with states, P_full, P_train, marginals, in_sample, holdout, summary.
    """
    df = df[df[year_col] == target_year].copy()
    if len(df) == 0:
        raise ValueError(f"No data for year {target_year}.")

    states = sorted(df[state_col].dropna().unique().tolist())
    quarters = sorted(df[quarter_col].dropna().unique().tolist())
    marginals = {
        q: _empirical_marginal(df, target_year, q, states, year_col, quarter_col, state_col)
        for q in quarters
    }

    all_pairs = [(q, q + 1) for q in quarters[:-1] if q + 1 in quarters]
    trans_full = pd.concat([
        _build_pair_transitions(df, p, id_col, year_col, quarter_col, state_col, interview_col, target_year)
        for p in all_pairs
    ], ignore_index=True)
    P_full, _ = _matrix_from_transitions(trans_full, states)

    trans_train = pd.concat([
        _build_pair_transitions(df, p, id_col, year_col, quarter_col, state_col, interview_col, target_year)
        for p in holdout_train_pairs
    ], ignore_index=True)
    P_train, _ = _matrix_from_transitions(trans_train, states)

    # In-sample propagation
    in_sample: dict = {}
    pi_chain = marginals[initial_quarter].copy()
    mass_lost_per_step: list = []
    prev_emp = marginals[initial_quarter]

    for k in horizons:
        pi_chain = pi_chain @ P_full
        step_mass = float(pi_chain.sum())
        mass_lost_per_step.append(1.0 - step_mass)
        if renormalize_after_step and step_mass > 0:
            pi_chain = pi_chain / step_mass
        target_q = initial_quarter + k
        if target_q not in marginals:
            continue
        emp = marginals[target_q]
        in_sample[k] = {
            'target_quarter': target_q,
            'pred': pi_chain.copy(), 'emp': emp.copy(),
            'metrics': _metrics(pi_chain, emp),
            'baseline_no_change': _metrics(prev_emp, emp),
            'mass_lost_cumulative': float(sum(mass_lost_per_step)),
        }
        prev_emp = emp

    # Hold-out
    test_start_q, test_end_q = holdout_test_pair
    pi_start = marginals[test_start_q]
    pi_pred_ho = pi_start @ P_train
    mass_ho = float(pi_pred_ho.sum())
    if renormalize_after_step and mass_ho > 0:
        pi_pred_ho = pi_pred_ho / mass_ho
    emp_ho = marginals[test_end_q]
    trans_test = _build_pair_transitions(
        df, holdout_test_pair, id_col, year_col, quarter_col, state_col, interview_col, target_year
    )
    holdout = {
        'test_start_quarter': test_start_q, 'test_end_quarter': test_end_q,
        'train_pairs': list(holdout_train_pairs),
        'pred': pi_pred_ho, 'emp': emp_ho,
        'metrics': _metrics(pi_pred_ho, emp_ho),
        'baseline_no_change': _metrics(pi_start, emp_ho),
        'mass_lost': 1.0 - mass_ho,
        'n_train_transitions': int(len(trans_train)),
        'n_test_pair_transitions': int(len(trans_test)),
    }

    rows = []
    for k, d in in_sample.items():
        rows.append({
            'test': f'in-sample k={k}', 'from_q': initial_quarter, 'to_q': d['target_quarter'],
            'tvd_model': d['metrics']['tvd'], 'tvd_baseline': d['baseline_no_change']['tvd'],
            'mae_model': d['metrics']['mae'], 'corr': d['metrics']['corr'],
            'mass_lost_cum': d['mass_lost_cumulative'],
        })
    rows.append({
        'test': 'hold-out', 'from_q': test_start_q, 'to_q': test_end_q,
        'tvd_model': holdout['metrics']['tvd'], 'tvd_baseline': holdout['baseline_no_change']['tvd'],
        'mae_model': holdout['metrics']['mae'], 'corr': holdout['metrics']['corr'],
        'mass_lost_cum': holdout['mass_lost'],
    })
    summary = pd.DataFrame(rows)

    if verbose:
        print("predictive_validation:")
        print(f"  Target year              : {target_year}")
        print(f"  Quarters observed        : {quarters}")
        print(f"  State space size         : {len(states)}")
        print(f"  P_full transitions       : {len(trans_full):,}")
        print(f"  P_train pairs            : {list(holdout_train_pairs)}")
        print(f"  P_train transitions      : {len(trans_train):,}")
        print(f"  Hold-out test pair       : {holdout_test_pair}  ({holdout['n_test_pair_transitions']:,} transitions)")
        print(f"  Renormalize after step   : {renormalize_after_step}")
        print()
        print("  Summary (TVD: lower is better; compare to baseline):")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return {
        'states': states, 'P_full': P_full, 'P_train': P_train,
        'marginals': marginals, 'in_sample': in_sample,
        'holdout': holdout, 'summary': summary, 'target_year': target_year,
    }
