"""
stationary.py
=============
Stationary distribution analysis for the deprivation-state Markov chain.

Computes the long-run (stationary) probability vector π, verifies ergodicity,
derives first-passage (hitting) times and mean recurrence times, quantifies
equilibrium flux (turnover / half-life), and identifies the two structural
traps described in the paper.

Structural traps
----------------
Low-Qualification Trap (LQT)
    States with ≥2 deprivations AND (d1=1 OR d2=1).
    Characterised by slow resolution of human-capital deficits.

Unprotection Trap (UT)
    States with ≥2 deprivations AND d4=1 AND d5=1.
    Co-occurrence of informality and social-security absence.

Bit convention: d1 d2 d3 d4 d5, MSB-first ('01011' means d2, d4, d5 active).

Public entry points
-------------------
check_ergodic(P)
    Verify irreducibility (strongly connected) and aperiodicity (self-loop).

stationary_distribution(P, cross_check=True)
    Compute π via eigendecomposition; cross-check with power iteration.

verify_stationary(P, pi)
    Assert πP ≈ π and Σπ ≈ 1.

hitting_times(P, target_indices)
    Solve (I − Q) h = 1 for expected first-passage times to a target set.

mean_recurrence_times(pi)
    Return 1/π_i for each state.

turnover_stats(P, pi, subset_indices)
    Compute equilibrium inflow, outflow, turnover rate, and half-life for
    a subset of states.

analyze_traps(states, pi, empirical=None, P=None)
    Summarise both traps, their overlap, and their union, with optional
    dynamics (requires P).

report(P, counts)
    Print a full ergodicity + π + trap analysis report.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

# Bit positions for each indicator (MSB-first, index into the 5-char string)
_D1, _D2, _D3, _D4, _D5 = 0, 1, 2, 3, 4
_NON_DEPRIVED = '00000'
_STATES = [format(i, '05b') for i in range(32)]


# ---------------------------------------------------------------------------
# Ergodicity checks
# ---------------------------------------------------------------------------

def check_ergodic(P: np.ndarray) -> dict:
    """Check irreducibility and aperiodicity of the transition matrix.

    Irreducibility: the directed graph (P > 0) has a single strongly
    connected component (SCC).

    Aperiodicity: at least one diagonal entry is positive (sufficient
    condition for an irreducible chain).

    Parameters
    ----------
    P : ndarray (S, S)
        Row-stochastic transition matrix.

    Returns
    -------
    dict with keys: irreducible, n_scc, aperiodic, ergodic.
    """
    A = (P > 0).astype(int)
    n_scc, _ = connected_components(A, directed=True, connection='strong')
    irreducible = (n_scc == 1)
    has_self_loop = bool((np.diag(P) > 0).any())
    return {
        'irreducible': irreducible,
        'n_scc':       int(n_scc),
        'aperiodic':   has_self_loop,
        'ergodic':     irreducible and has_self_loop,
    }


# ---------------------------------------------------------------------------
# Stationary distribution
# ---------------------------------------------------------------------------

def stationary_eigenvector(P: np.ndarray) -> np.ndarray:
    """Compute π via left eigenvector of P associated with eigenvalue 1."""
    w, V = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(V[:, idx])
    pi = np.abs(pi)
    return pi / pi.sum()


def stationary_power_iteration(
    P: np.ndarray, max_iter: int = 20_000, tol: float = 1e-14
) -> np.ndarray:
    """Compute π via power iteration: π_{n+1} = π_n P."""
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(max_iter):
        nxt = pi @ P
        if np.abs(nxt - pi).max() < tol:
            return nxt
        pi = nxt
    return pi


def stationary_distribution(
    P: np.ndarray,
    method: str = 'eigenvector',
    cross_check: bool = True,
    tol: float = 1e-8,
) -> np.ndarray:
    """Return the stationary distribution π of P.

    Parameters
    ----------
    P : ndarray (S, S)
        Row-stochastic transition matrix.
    method : str
        Primary computation method: 'eigenvector' or 'power_iteration'.
    cross_check : bool
        If True, compute by both methods and assert max|diff| < tol.
    tol : float
        Tolerance for the cross-check.

    Returns
    -------
    pi : ndarray (S,)
        Normalised stationary probability vector.
    """
    pi_eig = stationary_eigenvector(P)
    if cross_check:
        pi_pow = stationary_power_iteration(P)
        diff = float(np.abs(pi_eig - pi_pow).max())
        assert diff < tol, f"Methods disagree: max|diff| = {diff:.2e}"
    return pi_eig if method == 'eigenvector' else stationary_power_iteration(P)


def verify_stationary(P: np.ndarray, pi: np.ndarray, tol: float = 1e-10) -> None:
    """Assert that π is a valid stationary distribution: πP = π and Σπ = 1."""
    assert abs(pi.sum() - 1.0) < tol, f"Σπ = {pi.sum()}"
    resid = float(np.abs(pi @ P - pi).max())
    assert resid < tol, f"πP ≠ π: max|resid| = {resid:.2e}"


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def hitting_times(P: np.ndarray, target_indices: list) -> np.ndarray:
    """Compute expected first-passage times to a target set.

    For states not in the target set, solves:
        h_i = 1 + Σ_{j ∉ target} P_{ij} h_j,
    which in matrix form is (I − Q) h = 1.

    h[i] = 0 for i in target (already at the target).

    Parameters
    ----------
    P : ndarray (S, S)
        Row-stochastic transition matrix.
    target_indices : list of int
        State indices of the target set (e.g. [0] for '00000').

    Returns
    -------
    h : ndarray (S,)
        Expected first-passage time in number of periods (quarters).
    """
    n = P.shape[0]
    target = set(target_indices)
    assert target, "target_indices must be non-empty."
    non_target = [i for i in range(n) if i not in target]
    if not non_target:
        return np.zeros(n)

    Q = P[np.ix_(non_target, non_target)]
    try:
        h_nt = np.linalg.solve(np.eye(len(non_target)) - Q, np.ones(len(non_target)))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"(I−Q) is singular — target may be unreachable. ({exc})")

    h = np.zeros(n)
    for k, i in enumerate(non_target):
        h[i] = h_nt[k]
    return h


def mean_recurrence_times(pi: np.ndarray) -> np.ndarray:
    """Return 1/π_i for each state (expected return time, in periods).

    States with π_i = 0 return np.inf.
    """
    pi = np.asarray(pi, dtype=float)
    with np.errstate(divide='ignore'):
        return np.where(pi > 0, 1.0 / np.where(pi > 0, pi, 1.0), np.inf)


def turnover_stats(P: np.ndarray, pi: np.ndarray, subset_indices: list) -> dict:
    """Compute equilibrium flux statistics for a subset of states.

    At stationarity, inflow = outflow (detailed balance at the subset level).

    Parameters
    ----------
    P : ndarray (S, S)
    pi : ndarray (S,)
        Stationary distribution.
    subset_indices : list of int
        Indices of the states forming the subset S.

    Returns
    -------
    dict with:
        mass          : Σ_{i ∈ S} π_i
        outflow       : Σ_{i ∈ S, j ∉ S} π_i P_{ij}
        inflow        : Σ_{i ∉ S, j ∈ S} π_i P_{ij}
        turnover_rate : outflow / mass
        half_life     : ln(2) / −ln(1 − turnover_rate), in periods
    """
    n = P.shape[0]
    S_set = set(subset_indices)
    not_S = [i for i in range(n) if i not in S_set]

    mass = float(sum(pi[i] for i in S_set))
    outflow = float(sum(pi[i] * P[i, j] for i in S_set for j in not_S))
    inflow = float(sum(pi[i] * P[i, j] for i in not_S for j in S_set))
    rate = (outflow / mass) if mass > 0 else 0.0

    if 0 < rate < 1:
        half_life = math.log(2) / -math.log(1 - rate)
    elif rate == 0:
        half_life = float('inf')
    else:
        half_life = 0.0

    return {
        'mass': mass, 'outflow': outflow, 'inflow': inflow,
        'turnover_rate': rate, 'half_life': half_life,
    }


# ---------------------------------------------------------------------------
# Structural trap predicates
# ---------------------------------------------------------------------------

def low_qualification_predicate(state: str) -> bool:
    """True if state has ≥2 deprivations AND (d1=1 OR d2=1)."""
    return state.count('1') >= 2 and (state[_D1] == '1' or state[_D2] == '1')


def unprotection_predicate(state: str) -> bool:
    """True if state has ≥2 deprivations AND d4=1 AND d5=1."""
    return state.count('1') >= 2 and state[_D4] == '1' and state[_D5] == '1'


def _mass_of_subset(states: list, vec: np.ndarray, state_subset: list) -> float:
    """Return Σ vec[i] for i in state_subset."""
    idx = [states.index(s) for s in state_subset]
    return float(sum(vec[i] for i in idx))


# ---------------------------------------------------------------------------
# Trap analysis
# ---------------------------------------------------------------------------

def analyze_traps(
    states: list,
    pi: np.ndarray,
    empirical: np.ndarray | None = None,
    P: np.ndarray | None = None,
) -> dict:
    """Analyse both structural traps, their overlap, and union.

    Parameters
    ----------
    states : list of str
        Ordered state labels ('00000'…'11111').
    pi : ndarray (S,)
        Stationary distribution.
    empirical : ndarray (S,) or None
        Empirical origin-count distribution (for comparing π with the data).
    P : ndarray (S, S) or None
        If provided, adds dynamics: turnover, half-life, and weighted
        hitting time to '00000'.

    Returns
    -------
    dict with keys: low_qualification, unprotection, overlap, union.
    Each value is a dict with states, pi_mass, and optionally emp_mass,
    inflow, outflow, turnover_rate, half_life, avg_hitting_to_00000.
    """
    lqt = [s for s in states if low_qualification_predicate(s)]
    upt = [s for s in states if unprotection_predicate(s)]
    overlap = sorted(set(lqt) & set(upt))
    union = sorted(set(lqt) | set(upt))

    h_to_nd = hitting_times(P, [states.index(_NON_DEPRIVED)]) if P is not None else None

    def _block(state_list: list) -> dict:
        d: dict = {
            'states':   state_list,
            'pi_mass':  _mass_of_subset(states, pi, state_list),
        }
        if empirical is not None:
            d['emp_mass'] = _mass_of_subset(states, empirical, state_list)
        if P is not None and state_list:
            idx = [states.index(s) for s in state_list]
            stats_d = turnover_stats(P, pi, idx)
            d.update({
                'inflow':              stats_d['inflow'],
                'outflow':             stats_d['outflow'],
                'turnover_rate':       stats_d['turnover_rate'],
                'half_life':           stats_d['half_life'],
            })
            pi_sub = np.array([pi[i] for i in idx])
            w = pi_sub / pi_sub.sum() if pi_sub.sum() > 0 else pi_sub
            d['avg_hitting_to_00000'] = float(np.dot(w, [h_to_nd[i] for i in idx]))
        return d

    return {
        'low_qualification': _block(lqt),
        'unprotection':      _block(upt),
        'overlap':           _block(overlap),
        'union':             _block(union),
    }


# ---------------------------------------------------------------------------
# I/O helpers and full report
# ---------------------------------------------------------------------------

def load_matrix(P_df: pd.DataFrame) -> tuple[np.ndarray, list]:
    """Validate and extract a 32×32 P matrix from a DataFrame.

    Accepts either a 32×32 DataFrame (columns = state strings) or a 32×33
    DataFrame where the first column is a state-label column.

    Returns
    -------
    P : ndarray (32, 32)
    states : list of 32 state strings in column order.
    """
    if P_df.shape == (32, 33):
        P_df = P_df.iloc[:, 1:]
    assert P_df.shape == (32, 32), f"Expected 32×32, got {P_df.shape}"
    assert list(P_df.columns) == _STATES, "Column order does not match '00000'…'11111'"
    P = P_df.values.astype(float)
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), \
        f"Rows do not sum to 1: range [{row_sums.min()}, {row_sums.max()}]"
    return P, _STATES


def empirical_from_counts(counts_df: pd.DataFrame) -> np.ndarray:
    """Derive the empirical origin-state distribution from a counts matrix.

    Returns N_i / N_total where N_i = Σ_j N_{ij} (transitions out of state i).
    """
    M = counts_df.values.astype(float)
    N = M.sum(axis=1)
    return N / N.sum()


def report(P_df: pd.DataFrame, counts_df: pd.DataFrame) -> dict:
    """Print a complete ergodicity, π, and structural-trap report.

    Parameters
    ----------
    P_df : pd.DataFrame
        32×32 transition probability matrix.
    counts_df : pd.DataFrame
        32×32 transition count matrix.

    Returns
    -------
    dict with P, pi, ergodicity, empirical, hitting_to_00000,
    mean_recurrence, traps.
    """
    P, states = load_matrix(P_df)
    erg = check_ergodic(P)
    assert erg['ergodic'], f"Chain is not ergodic: {erg}"

    pi = stationary_distribution(P, cross_check=True)
    verify_stationary(P, pi)
    emp = empirical_from_counts(counts_df)
    h_nd = hitting_times(P, [0])
    m_rec = mean_recurrence_times(pi)

    print("Ergodicity:", erg)
    print(f"\nπ:  Σ={pi.sum():.6f}  min={pi.min():.6f}  max={pi.max():.4f}")
    print(f"π['00000'] = {pi[0]:.4f}")

    print("\nTop 10 states by π:")
    order = np.argsort(-pi)
    header = f"   {'state':>8s} {'#depriv':>7s} {'π':>10s} {'empirical':>10s} {'π/emp':>8s} {'h→00000':>10s} {'1/π':>10s}"
    print(header)
    for k in order[:10]:
        print(
            f"   {states[k]:>8s} {states[k].count('1'):>7d} {pi[k]:>10.4f}"
            f" {emp[k]:>10.4f} {pi[k] / emp[k] if emp[k] > 0 else float('nan'):>8.2f}"
            f" {h_nd[k]:>10.2f} {m_rec[k]:>10.1f}"
        )

    print("\nStructural traps (mass + dynamics):")
    traps = analyze_traps(states, pi, emp, P=P)
    for name in ('low_qualification', 'unprotection', 'overlap', 'union'):
        t = traps[name]
        line = (
            f"   {name:>18s}: {len(t['states']):>2d} profiles  "
            f"π={t['pi_mass'] * 100:5.2f}%"
        )
        if 'emp_mass' in t:
            line += f"  emp={t['emp_mass'] * 100:5.2f}%"
        if 'turnover_rate' in t:
            line += (
                f"  turnover={t['turnover_rate'] * 100:5.1f}%/q"
                f"  half_life={t['half_life']:4.1f}q"
                f"  h→00000={t['avg_hitting_to_00000']:5.1f}q"
            )
        print(line)

    return {
        'P': P, 'pi': pi, 'ergodicity': erg, 'empirical': emp,
        'hitting_to_00000': h_nd, 'mean_recurrence': m_rec, 'traps': traps,
    }
