"""
nk_core.py — Numba-JIT accelerated Monte Carlo simulation on NK fitness landscapes.

This module implements the core evolutionary dynamics engine used in:
    "Three dynamical regimes on high-dimensional fitness landscapes:
     the vanishing punctuated band and the deep-basin condition"

Model:
    - Binary genotype g ∈ {0,1}^d on an NK fitness landscape with Gaussian basins
    - Single-lineage Metropolis dynamics (weak-mutation limit, Nμ ≪ 1)
    - One random locus flipped per step; accepted with prob min(1, exp(scoeff · Δw))

Design:
    - All random numbers pre-generated with numpy (reliable MT19937)
    - Inner loop compiled with Numba JIT (fast, no Python overhead)
    - Thread-safe: each call to run_single_mc() is fully independent
"""

import numpy as np
import numba as nb
from numba import njit, int32, int64, float64
from dataclasses import dataclass
from typing import Optional
import time


# ============================================================
# Data structures
# ============================================================

@dataclass
class MCParams:
    """Parameters for a single Monte Carlo simulation run."""
    d: int               # Number of loci (genetic dimensionality)
    K: int               # Epistasis parameter (K nearest-neighbor interactions)
    beta: float          # Basin depth (tunable fitness bonus amplitude)
    scoeff: float        # Dimensionless selection strength
    n_steps: int         # Number of mutation steps to simulate
    seed: int            # Random seed for reproducibility
    h_star: float = 0.35         # Basin boundary threshold (Hamming fraction)
    record_dw: bool = False      # Whether to record fitness differentials
    dw_max_samples: int = 50000  # Max Δw samples to store
    record_trajectory: bool = False  # Whether to record ḡ(t) trajectory
    traj_interval: int = 1000       # Steps between trajectory samples


@dataclass
class MCResult:
    """Results from a single Monte Carlo simulation run."""
    d: int; K: int; beta: float; scoeff: float
    n_steps: int; seed: int; h_star: float
    n_transitions: int = 0       # Number of complete basin-to-basin transitions
    mean_stasis: float = 0.0     # Mean stasis duration (steps in basin)
    mean_transit: float = 0.0    # Mean transit duration (steps in transitional zone)
    ratio: float = 0.0           # Stasis/transit ratio
    stasis_durations: Optional[np.ndarray] = None
    transit_durations: Optional[np.ndarray] = None
    dw_mean: float = 0.0
    dw_std: float = 0.0
    dw_abs_mean: float = 0.0
    dw_abs_median: float = 0.0
    dw_n_samples: int = 0
    dw_samples: Optional[np.ndarray] = None
    trajectory_gbar: Optional[np.ndarray] = None
    trajectory_fitness: Optional[np.ndarray] = None
    wall_time: float = 0.0


# ============================================================
# JIT-compiled core functions
# ============================================================

@njit(cache=True)
def _build_neighbors(d, K):
    """Build K nearest-neighbor interaction table for NK model.

    For locus i, neighbors are (i+1)%d, (i+2)%d, ..., (i+K)%d.
    This is the standard circular neighbor topology.
    """
    neighbors = np.empty((d, max(K, 1)), dtype=int32)
    for i in range(d):
        for j in range(K):
            neighbors[i, j] = int32((i + j + 1) % d)
    return neighbors


@njit(cache=True)
def _nk_fitness(g, d, K, neighbors, fi_flat, table_size):
    """Compute NK fitness component: (1/d) Σᵢ fᵢ(gᵢ, g_neighbors).

    Args:
        g: Binary genotype array of length d
        fi_flat: Flattened lookup table of shape (d * 2^(K+1),)
        table_size: 2^(K+1), number of entries per locus
    """
    total = 0.0
    for i in range(d):
        idx = int32(g[i])
        for k in range(K):
            idx = idx * int32(2) + int32(g[neighbors[i, k]])
        total += fi_flat[i * table_size + idx]
    return total / float64(d)


@njit(cache=True)
def _basin_bonus(gbar, beta):
    """Compute Gaussian basin bonus: β · max(exp(-2ḡ²), exp(-2(1-ḡ)²)).

    Creates two symmetric fitness wells centered at ḡ=0 (all-zeros)
    and ḡ=1 (all-ones), each with depth β.
    """
    b1 = np.exp(-2.0 * gbar * gbar)
    b2 = np.exp(-2.0 * (1.0 - gbar) * (1.0 - gbar))
    if b1 > b2:
        return beta * b1
    return beta * b2


@njit(cache=True)
def _calc_gbar(g, d):
    """Compute Hamming fraction ḡ = (1/d) Σᵢ gᵢ."""
    s = int32(0)
    for i in range(d):
        s += g[i]
    return float64(s) / float64(d)


@njit(cache=True)
def _full_fitness(g, d, K, neighbors, fi_flat, table_size, beta):
    """Compute total fitness w(g) = NK_component + basin_bonus."""
    gbar = _calc_gbar(g, d)
    nk = _nk_fitness(g, d, K, neighbors, fi_flat, table_size)
    bb = _basin_bonus(gbar, beta)
    return nk + bb, gbar


@njit(cache=True)
def _mc_core(d, K, beta, scoeff, h_star, n_steps,
             neighbors, fi_flat, table_size,
             rand_loci, rand_accept, rand_init_flips,
             record_dw_interval, traj_interval,
             max_dw_samples, max_traj_points):
    """Core Monte Carlo loop with pre-generated random numbers.

    Implements single-lineage Metropolis dynamics on an NK landscape
    with two Gaussian fitness basins.

    Transition tracking:
        - Basin 1: ḡ < h*
        - Basin 2: ḡ > 1 - h*
        - Transitional: h* ≤ ḡ ≤ 1 - h*
        - Complete transition: basin X → transitional → basin Y (X ≠ Y)
        - Incomplete excursion: basin X → transitional → basin X (not counted)

    Args:
        rand_loci: int32[n_steps] — which locus to flip each step
        rand_accept: float64[n_steps] — uniform random for Metropolis
        rand_init_flips: int32[:] — initial loci to set to 1

    Returns:
        Tuple of (stasis_durations, transit_durations, dw_samples,
                  trajectory_gbar, trajectory_fitness, n_transitions, n_accepted)
    """
    # Initialize genotype (mostly zeros, with a few bits flipped)
    g = np.zeros(d, dtype=int32)
    for i in range(len(rand_init_flips)):
        g[rand_init_flips[i]] = int32(1)

    w, gbar = _full_fitness(g, d, K, neighbors, fi_flat, table_size, beta)

    # Classify initial state: 0=basin1, 1=transitional, 2=basin2
    if gbar < h_star:
        state = int32(0)
    elif gbar > 1.0 - h_star:
        state = int32(2)
    else:
        state = int32(1)

    # Pre-allocate output arrays
    max_tr = min(n_steps, 2000000)
    stasis_arr = np.empty(max_tr, dtype=int64)
    transit_arr = np.empty(max_tr, dtype=int64)
    n_tr = int64(0)

    dw_arr = np.empty(max_dw_samples, dtype=float64)
    dw_count = int64(0)

    traj_gb = np.empty(max_traj_points, dtype=float64)
    traj_w = np.empty(max_traj_points, dtype=float64)
    traj_count = int64(0)

    # Transition state machine
    last_basin = state if state != int32(1) else int32(-1)
    basin_entry = int64(0)
    transit_entry = int64(0)
    in_transit = False
    n_accepted = int64(0)

    # --- Main loop ---
    for step in range(n_steps):
        # Propose mutation: flip one random locus
        locus = rand_loci[step]
        old_val = g[locus]
        new_val = int32(1) - old_val
        g[locus] = new_val

        # Compute new fitness
        w_new, gbar_new = _full_fitness(
            g, d, K, neighbors, fi_flat, table_size, beta
        )
        dw = w_new - w

        # Record Δw sample (periodic)
        if record_dw_interval > 0 and dw_count < max_dw_samples:
            if step % record_dw_interval == 0:
                dw_arr[dw_count] = dw
                dw_count += 1

        # Metropolis acceptance
        accept = False
        if dw >= 0.0:
            accept = True
        elif rand_accept[step] < np.exp(scoeff * dw):
            accept = True

        if accept:
            w = w_new
            gbar = gbar_new
            n_accepted += 1
            if gbar < h_star:
                state = int32(0)
            elif gbar > 1.0 - h_star:
                state = int32(2)
            else:
                state = int32(1)
        else:
            g[locus] = old_val  # Revert mutation

        # Trajectory recording (periodic)
        if traj_interval > 0 and traj_count < max_traj_points:
            if step % traj_interval == 0:
                traj_gb[traj_count] = gbar
                traj_w[traj_count] = w
                traj_count += 1

        # --- Transition tracking state machine ---
        if not in_transit:
            if state == int32(1):
                # Entered transitional zone from a basin
                in_transit = True
                transit_entry = int64(step)
            elif state != int32(1):
                if last_basin == int32(-1):
                    # First time entering a basin
                    last_basin = state
                    basin_entry = int64(step)
                elif state != last_basin:
                    # Direct basin switch without transiting (rare at d≥10)
                    if n_tr < max_tr:
                        stasis_arr[n_tr] = int64(step) - basin_entry
                        transit_arr[n_tr] = int64(1)
                        n_tr += 1
                    last_basin = state
                    basin_entry = int64(step)
        else:
            if state != int32(1):
                # Exited transitional zone
                td = int64(step) - transit_entry
                if state != last_basin and last_basin >= int32(0):
                    # Complete transition: basin X → transit → basin Y
                    if n_tr < max_tr:
                        stasis_arr[n_tr] = transit_entry - basin_entry
                        transit_arr[n_tr] = td
                        n_tr += 1
                # Update basin tracking (even for incomplete excursions)
                last_basin = state
                basin_entry = int64(step)
                in_transit = False

    return (stasis_arr[:n_tr], transit_arr[:n_tr],
            dw_arr[:dw_count], traj_gb[:traj_count], traj_w[:traj_count],
            n_tr, n_accepted)


# ============================================================
# Python wrapper
# ============================================================

def run_single_mc(params: MCParams) -> MCResult:
    """Run one Monte Carlo simulation. Thread-safe and fully self-contained.

    Each call creates its own RNG, NK landscape, and genotype.
    Safe for use with multiprocessing.Pool.
    """
    t0 = time.perf_counter()
    d, K = params.d, params.K
    rng = np.random.RandomState(params.seed)

    # Build NK landscape
    neighbors = _build_neighbors(d, K) if K > 0 else np.zeros((d, 1), dtype=np.int32)
    table_size = 1 << (K + 1)
    fi_flat = rng.random(d * table_size).astype(np.float64)

    # Pre-generate ALL random numbers (numpy RNG, then pass to JIT)
    n = params.n_steps
    rand_loci = rng.randint(0, d, size=n).astype(np.int32)
    rand_accept = rng.random(n).astype(np.float64)

    # Initial genotype: ~10% of bits set to 1 (starts near basin 1)
    n_flip = max(1, d // 10)
    rand_init = rng.choice(d, size=n_flip, replace=False).astype(np.int32)

    # Compute recording intervals
    dw_interval = max(1, n // params.dw_max_samples) if params.record_dw else 0
    traj_interval = params.traj_interval if params.record_trajectory else 0
    max_traj = n // max(traj_interval, 1) + 1 if traj_interval > 0 else 1

    # Run JIT-compiled core
    sd, td, dw_arr, traj_gb, traj_w, n_tr, n_acc = _mc_core(
        d, K, params.beta, params.scoeff, params.h_star, n,
        neighbors, fi_flat, int32(table_size),
        rand_loci, rand_accept, rand_init,
        int64(dw_interval), int64(traj_interval),
        int64(params.dw_max_samples), int64(max_traj)
    )
    n_tr = int(n_tr)

    # Compute summary statistics
    if n_tr > 0:
        ms = float(np.mean(sd))
        mt = float(np.mean(td))
        ratio = ms / mt if mt > 0 else 1e10
    else:
        ms = float(n)
        mt = 0.0
        ratio = 1e10

    result = MCResult(
        d=d, K=K, beta=params.beta, scoeff=params.scoeff,
        n_steps=n, seed=params.seed, h_star=params.h_star,
        n_transitions=n_tr,
        mean_stasis=ms, mean_transit=mt,
        ratio=min(ratio, 1e10),
        stasis_durations=sd.copy() if n_tr > 0 else None,
        transit_durations=td.copy() if n_tr > 0 else None,
        wall_time=time.perf_counter() - t0,
    )

    if params.record_dw and len(dw_arr) > 0:
        result.dw_mean = float(np.mean(dw_arr))
        result.dw_std = float(np.std(dw_arr))
        result.dw_abs_mean = float(np.mean(np.abs(dw_arr)))
        result.dw_abs_median = float(np.median(np.abs(dw_arr)))
        result.dw_n_samples = len(dw_arr)
        result.dw_samples = dw_arr.copy()

    if params.record_trajectory and len(traj_gb) > 0:
        result.trajectory_gbar = traj_gb.copy()
        result.trajectory_fitness = traj_w.copy()

    return result
