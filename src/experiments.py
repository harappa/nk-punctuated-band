"""
experiments.py — Experiment definitions for Exp 1–9.

Each experiment is defined as a function returning a dict with:
    - 'name': Human-readable description
    - 'jobs': List of MCParams (or PolyParams/WFParams) for parallel execution
    - 'worker': Optional string ('poly' or 'wf') for variant experiments

All experiments are registered in the EXPERIMENTS dict at module level.

Experiment overview:
    1: Phase diagram mapping (primary experiment, fine β resolution)
    2: Dimension precision scan (d=12–18, punctuated band vanishing point)
    3: Extended large-d runs (false-absence exclusion + slow-diffusion characterization)
    4: Epistasis sensitivity (K = 0, 2, 4, 8, 16)
    5: Selection strength (scoeff) scaling with d
    6: Regime classification threshold sensitivity (reanalysis only)
    7: Basin boundary (h*) sensitivity
    8: Alternative basin shapes (polynomial p=1.0, 1.5, 2.0 vs Gaussian)
    9: Population dynamics (Wright-Fisher, Nμ = 1, 10)
"""

import numpy as np
from typing import List, Dict, Any
from nk_core import MCParams


# ============================================================
# Simulation length constants
# ============================================================

STANDARD_N_STEPS = 5_000_000    # 5M steps per run (baseline)
STANDARD_N_REPS = 20            # 20 replicates per (d, K, β) point
EXTENDED_N_STEPS = 20_000_000   # 20M steps (rare-event testing)
EXTENDED_N_REPS = 50            # 50 replicates (robust CI)

# Quick validation mode (for CI/testing)
QUICK_N_STEPS = 200_000
QUICK_N_REPS = 3


# ============================================================
# Job generation helper
# ============================================================

def _make_jobs(d_list, K_list, beta_list, scoeff_list, n_steps, n_reps,
               record_dw=False, record_traj=False, h_star=0.35,
               base_seed=42) -> List[MCParams]:
    """Generate MCParams jobs for a parameter grid.

    Seeds are generated via a deterministic hash of all parameters,
    ensuring uniqueness across experiments and avoiding collisions
    from linear combinations.
    """
    import hashlib

    jobs = []
    for d in d_list:
        for K in K_list:
            if K >= d:
                continue
            for beta in beta_list:
                for scoeff in scoeff_list:
                    for rep in range(n_reps):
                        # Hash-based seed: deterministic, collision-free
                        key = f"{base_seed}:{rep}:{d}:{K}:{beta}:{scoeff}:{h_star}"
                        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                        jobs.append(MCParams(
                            d=d, K=K, beta=float(beta), scoeff=float(scoeff),
                            n_steps=n_steps, seed=seed, h_star=h_star,
                            record_dw=record_dw, record_trajectory=record_traj,
                        ))
    return jobs


# ============================================================
# Experiment definitions
# ============================================================

def exp_1(quick=False) -> Dict[str, Any]:
    """Exp 1: Phase diagram mapping (primary experiment).

    Fine-grained β grid (29 values) across 10 dimensions to map the
    full (d, β) parameter space and locate phase boundaries β*_low(d)
    and β*_high(d) precisely. 30 replicates per point.

    The β grid includes fine resolution at β = 3.0–6.0 in Δβ = 0.5
    increments to resolve the diffusive-to-punctuated transition at
    low d (particularly d = 10).
    """
    n_steps = QUICK_N_STEPS if quick else STANDARD_N_STEPS
    n_reps = QUICK_N_REPS if quick else 30
    beta_fine = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                 5.5, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0,
                 24.0, 28.0, 32.0, 40.0, 48.0, 56.0, 64.0, 80.0,
                 96.0, 128.0]
    return {
        'name': 'Phase diagram mapping (primary experiment)',
        'jobs': _make_jobs(
            d_list=[10, 20, 40, 60, 80, 100, 120, 150, 200, 300],
            K_list=[2],
            beta_list=beta_fine,
            scoeff_list=[5.0],
            n_steps=n_steps, n_reps=n_reps,
        ),
    }


def exp_2(quick=False) -> Dict[str, Any]:
    """Exp 2: Dimension precision scan (d = 12, 14, 16, 18).

    Locates the precise dimension at which the punctuated band
    vanishes, filling the gap between d=10 (punctuated) and d=20
    (absent). Provides quantitative d_eff upper bound.
    """
    n_steps = QUICK_N_STEPS if quick else STANDARD_N_STEPS
    n_reps = QUICK_N_REPS if quick else 30
    return {
        'name': 'Dimension precision scan (d=12-18)',
        'jobs': _make_jobs(
            d_list=[12, 14, 16, 18],
            K_list=[2],
            beta_list=[2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32],
            scoeff_list=[5.0],
            n_steps=n_steps, n_reps=n_reps,
            base_seed=300000,
        ),
    }


def exp_3(quick=False) -> Dict[str, Any]:
    """Exp 3: Extended runs at large d.

    Verifies that n_tr=0 at high d reflects true freezing, not
    insufficient simulation length. Uses 50 reps × 20M steps.
    Covers both high-β (false-absence exclusion) and low-β
    (slow-diffusion characterization at d=300 where transit times
    scale as ~d³ ≈ 2.7×10⁷, exceeding the primary simulation length).
    """
    n_steps = QUICK_N_STEPS if quick else EXTENDED_N_STEPS
    n_reps = QUICK_N_REPS if quick else EXTENDED_N_REPS
    return {
        'name': 'Extended large-d runs',
        'jobs': _make_jobs(
            d_list=[80, 150, 300],
            K_list=[2],
            beta_list=[0.5, 1.0, 2.0, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64],
            scoeff_list=[5.0],
            n_steps=n_steps, n_reps=n_reps,
        ),
    }


def exp_4(quick=False) -> Dict[str, Any]:
    """Exp 4: Epistasis sensitivity (K = 0, 2, 4, 8, 16).

    Tests whether the vanishing punctuated band persists at higher
    epistasis levels, ruling out artifacts from K=2 smoothness.
    """
    n_steps = QUICK_N_STEPS if quick else STANDARD_N_STEPS
    n_reps = QUICK_N_REPS if quick else STANDARD_N_REPS
    return {
        'name': 'Epistasis sensitivity (K=0,2,4,8,16)',
        'jobs': _make_jobs(
            d_list=[10, 20, 40, 80, 150],
            K_list=[0, 2, 4, 8, 16],
            beta_list=[0.5, 1, 2, 4, 8, 16, 32, 64],
            scoeff_list=[5.0],
            n_steps=n_steps, n_reps=n_reps,
        ),
    }


def exp_5(quick=False) -> Dict[str, Any]:
    """Exp 5: Selection strength scaling with d.

    Three scaling rules:
        - scoeff = 5.0 (fixed baseline)
        - scoeff = 5.0 × √(d/10)  (compensates NK variance ∝ 1/d)
        - scoeff = 5.0 × (d/10)   (strong stabilizing selection)
    """
    n_steps = QUICK_N_STEPS if quick else STANDARD_N_STEPS
    n_reps = QUICK_N_REPS if quick else STANDARD_N_REPS
    import hashlib
    jobs = []
    for d in [10, 20, 40, 80, 150]:
        for label, sc_val in [
            ("fixed", 5.0),
            ("sqrt_d", 5.0 * np.sqrt(d / 10.0)),
            ("linear_d", 5.0 * (d / 10.0)),
        ]:
            for beta in [0.5, 2, 4, 8, 16, 32, 64]:
                for rep in range(n_reps):
                    key = f"5:{rep}:{d}:{beta}:{sc_val}"
                    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                    jobs.append(MCParams(
                        d=d, K=2, beta=float(beta), scoeff=float(sc_val),
                        n_steps=n_steps, seed=seed,
                    ))
    return {
        'name': 'Selection strength scaling with d',
        'jobs': jobs,
    }


def exp_6(quick=False) -> Dict[str, Any]:
    """Exp 6: Threshold sensitivity (ratio = 5, 10, 20).

    No separate simulation needed — reanalysis of Exp 1 data
    with different regime classification thresholds.
    """
    return {
        'name': 'Threshold sensitivity (reanalysis of Exp 1)',
        'jobs': [],
    }


def exp_7(quick=False) -> Dict[str, Any]:
    """Exp 7: Basin boundary (h*) sensitivity.

    Tests whether results depend on the choice h*=0.35 by scanning
    h* ∈ {0.30, 0.35, 0.40}.
    """
    n_steps = QUICK_N_STEPS if quick else STANDARD_N_STEPS
    n_reps = QUICK_N_REPS if quick else STANDARD_N_REPS
    import hashlib
    jobs = []
    for h_star in [0.30, 0.35, 0.40]:
        for d in [10, 40, 80, 150]:
            for beta in [0.5, 2, 4, 8, 16, 32, 64]:
                for rep in range(n_reps):
                    key = f"7:{rep}:{d}:{beta}:{h_star}"
                    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                    jobs.append(MCParams(
                        d=d, K=2, beta=float(beta), scoeff=5.0,
                        n_steps=n_steps, seed=seed, h_star=h_star,
                    ))
    return {
        'name': 'Basin boundary (h*) sensitivity',
        'jobs': jobs,
    }


def exp_8(quick=False) -> Dict[str, Any]:
    """Exp 8: Alternative basin shapes.

    Compares Gaussian baseline (p_shape=-1) against polynomial basins
    (p=1.0, 1.5, 2.0). Tests whether the vanishing threshold d≈20 is
    specific to the Gaussian shape. Key finding: polynomial basins produce
    NO punctuated regime even at d=10, because they drop to zero bonus
    at the basin boundary, unlike Gaussian which retains 78%.
    """
    from nk_variants import PolyParams

    n_steps = QUICK_N_STEPS if quick else STANDARD_N_STEPS
    n_reps = QUICK_N_REPS if quick else 20
    import hashlib
    jobs = []
    for p_shape in [-1.0, 1.0, 1.5, 2.0]:
        for d in [10, 14, 18, 20, 40]:
            for beta in [2, 4, 6, 8, 12, 16, 32]:
                for rep in range(n_reps):
                    key = f"8:{rep}:{d}:{beta}:{p_shape}"
                    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                    jobs.append(PolyParams(
                        d=d, K=2, beta=float(beta), scoeff=5.0,
                        n_steps=n_steps, seed=seed, h_star=0.35,
                        p_shape=p_shape,
                    ))
    return {
        'name': 'Alternative basin shapes (Gaussian vs polynomial)',
        'jobs': jobs,
        'worker': 'poly',
    }


def exp_9(quick=False) -> Dict[str, Any]:
    """Exp 9: Population dynamics (Wright-Fisher).

    Three Nμ regimes to test whether population effects rescue or
    accelerate the vanishing of the punctuated band:
        N=10,  μ=0.1   → Nμ=1   (near single-lineage)
        N=100, μ=0.01  → Nμ=1   (larger pop, same mutation load)
        N=100, μ=0.1   → Nμ=10  (strong mutation, quasispecies regime)

    Note: uses generation-based timescale (not mutation steps), so
    quantitative comparison with Exp 1–5 requires timescale matching.
    """
    from nk_variants import WFParams

    n_gens = 10_000 if quick else 200_000
    n_reps = QUICK_N_REPS if quick else 10
    import hashlib
    jobs = []
    for N_pop, mu_rate in [(10, 0.1), (100, 0.01), (100, 0.1)]:
        for d in [10, 20, 40]:
            for beta in [2, 4, 8, 16, 32]:
                for rep in range(n_reps):
                    key = f"9:{rep}:{d}:{beta}:{N_pop}:{mu_rate}"
                    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                    jobs.append(WFParams(
                        d=d, K=2, beta=float(beta), scoeff=5.0,
                        n_steps=n_gens, seed=seed, h_star=0.35,
                        N_pop=N_pop, mu_rate=mu_rate,
                    ))
    return {
        'name': 'Population dynamics (Wright-Fisher, Nμ=1,10)',
        'jobs': jobs,
        'worker': 'wf',
    }


# ============================================================
# Experiment registry
# ============================================================

EXPERIMENTS = {
    '1': exp_1,
    '2': exp_2,
    '3': exp_3,
    '4': exp_4,
    '5': exp_5,
    '6': exp_6,
    '7': exp_7,
    '8': exp_8,
    '9': exp_9,
}
