"""
nk_variants.py — Alternative basin shapes and population dynamics.

Extends the core MC engine (nk_core.py) with:
    1. Polynomial basin shapes (Exp 8): configurable exponent p
    2. Wright-Fisher population dynamics (Exp 9): finite population with selection

Both variants reuse the NK fitness computation from nk_core.py.
"""

import numpy as np
import time
from numba import njit, int32, int64, float64
from dataclasses import dataclass
from typing import Optional, Dict, Any

from nk_core import _build_neighbors, _nk_fitness, _calc_gbar


# ============================================================
# Polynomial basin bonus
# ============================================================

@njit(cache=True)
def _basin_bonus_polynomial(gbar, beta, p, h_star):
    """Polynomial basin bonus with exponent p.

    Computes β × (1 - x)^p where x = min(ḡ, 1-ḡ) / h* is the normalized
    distance from the nearest basin center. The bonus equals β at basin
    centers (ḡ=0 or ḡ=1) and drops to zero at the basin boundary (ḡ=h*).

    Beyond the boundary (x > 1), a residual tail decays smoothly to zero
    at the midpoint (ḡ=0.5) to maintain continuity.

    For comparison, the Gaussian bonus retains 78% of its peak at the
    basin boundary (h*=0.35), while all polynomial forms drop to zero.
    """
    dist_to_center = min(gbar, 1.0 - gbar)
    x = dist_to_center / h_star
    if x >= 1.0:
        tail = 1.0 - (x - 1.0) / (1.0 / h_star - 1.0)
        if tail <= 0.0:
            return 0.0
        return beta * (tail ** p)
    return beta * ((1.0 - x) ** p)


@njit(cache=True)
def _basin_bonus_gaussian(gbar, beta):
    """Original Gaussian basin bonus (baseline for comparison)."""
    b1 = np.exp(-2.0 * gbar * gbar)
    b2 = np.exp(-2.0 * (1.0 - gbar) * (1.0 - gbar))
    return beta * max(b1, b2)


# ============================================================
# MC core with configurable basin shape
# ============================================================

@njit(cache=True)
def _mc_core_poly(d, K, beta, scoeff, h_star, n_steps, p_shape,
                  neighbors, fi_flat, table_size,
                  rand_loci, rand_accept, rand_init_flips):
    """MC core with polynomial or Gaussian basin shape.

    Args:
        p_shape: Basin shape exponent. Use -1 for Gaussian (baseline).
                 Positive values use polynomial form with that exponent.

    Returns:
        (stasis_durations, transit_durations, n_transitions)
    """
    # Initialize genotype
    g = np.zeros(d, dtype=int32)
    for i in range(len(rand_init_flips)):
        g[rand_init_flips[i]] = int32(1)

    gbar = _calc_gbar(g, d)
    nk = _nk_fitness(g, d, K, neighbors, fi_flat, table_size)
    if p_shape < 0:
        bb = _basin_bonus_gaussian(gbar, beta)
    else:
        bb = _basin_bonus_polynomial(gbar, beta, p_shape, h_star)
    w = nk + bb

    # Classify initial state
    if gbar < h_star:
        state = int32(0)
    elif gbar > 1.0 - h_star:
        state = int32(2)
    else:
        state = int32(1)

    # Output arrays
    max_tr = min(n_steps, 2000000)
    stasis_arr = np.empty(max_tr, dtype=int64)
    transit_arr = np.empty(max_tr, dtype=int64)
    n_tr = int64(0)

    # State machine
    last_basin = state if state != int32(1) else int32(-1)
    basin_entry = int64(0)
    transit_entry = int64(0)
    in_transit = False

    for step in range(n_steps):
        locus = rand_loci[step]
        old_val = g[locus]
        g[locus] = int32(1) - old_val

        gbar_new = _calc_gbar(g, d)
        nk_new = _nk_fitness(g, d, K, neighbors, fi_flat, table_size)
        if p_shape < 0:
            bb_new = _basin_bonus_gaussian(gbar_new, beta)
        else:
            bb_new = _basin_bonus_polynomial(gbar_new, beta, p_shape, h_star)
        w_new = nk_new + bb_new
        dw = w_new - w

        accept = False
        if dw >= 0.0:
            accept = True
        elif rand_accept[step] < np.exp(scoeff * dw):
            accept = True

        if accept:
            w = w_new
            gbar = gbar_new
            if gbar < h_star:
                state = int32(0)
            elif gbar > 1.0 - h_star:
                state = int32(2)
            else:
                state = int32(1)
        else:
            g[locus] = old_val

        # Transition tracking (identical to nk_core._mc_core)
        step64 = int64(step)
        if not in_transit:
            if state == int32(1):
                in_transit = True
                transit_entry = step64
            elif state != int32(1):
                if last_basin == int32(-1):
                    last_basin = state
                    basin_entry = step64
                elif state != last_basin:
                    if n_tr < max_tr:
                        stasis_arr[n_tr] = step64 - basin_entry
                        transit_arr[n_tr] = int64(1)
                        n_tr += 1
                    last_basin = state
                    basin_entry = step64
        else:
            if state != int32(1):
                td = step64 - transit_entry
                if state != last_basin and last_basin >= int32(0):
                    if n_tr < max_tr:
                        stasis_arr[n_tr] = transit_entry - basin_entry
                        transit_arr[n_tr] = td
                        n_tr += 1
                last_basin = state
                basin_entry = step64
                in_transit = False

    return stasis_arr[:n_tr], transit_arr[:n_tr], n_tr


# ============================================================
# Wright-Fisher population dynamics
# ============================================================

@njit(cache=True)
def _mc_core_wf(d, K, beta, scoeff, h_star, n_gens, N_pop, mu_rate,
                neighbors, fi_flat, table_size, rand_init_flips, seed):
    """Wright-Fisher population dynamics with fitness-proportional selection.

    Each generation:
        1. Mutation: each locus flips with probability mu_rate
        2. Fitness evaluation for all individuals
        3. Selection: fitness-proportional sampling (Wright-Fisher)

    The population centroid ḡ_mean is used for regime classification,
    following the same state machine as single-lineage dynamics.

    Unlike the single-lineage MC core (nk_core.py), which pre-generates
    all random numbers via numpy and passes them to JIT, the WF
    implementation uses Numba's internal np.random within the JIT
    function. This is because pre-generating mutation arrays would
    require N_pop × d × n_gens floats (e.g., 100 × 40 × 200000
    = 800M entries, ~6 GB), which is impractical. Reproducibility
    is ensured by seeding the Numba-internal RNG at the start of
    this function via np.random.seed(seed).

    Args:
        N_pop: Population size (number of haploid individuals)
        mu_rate: Per-locus mutation probability per generation
        n_gens: Number of generations to simulate
        seed: Random seed for Numba-internal RNG (ensures reproducibility)

    Returns:
        (stasis_durations, transit_durations, n_transitions)
    """
    # Seed Numba's internal RNG for reproducibility
    np.random.seed(seed)

    # Initialize population (all start at same genotype)
    pop = np.zeros((N_pop, d), dtype=int32)
    for ind in range(N_pop):
        for i in range(len(rand_init_flips)):
            pop[ind, rand_init_flips[i]] = int32(1)

    # Output arrays
    max_tr = min(n_gens, 500000)
    stasis_arr = np.empty(max_tr, dtype=int64)
    transit_arr = np.empty(max_tr, dtype=int64)
    n_tr = int64(0)

    # Initial centroid state
    gbar_sum = 0.0
    for ind in range(N_pop):
        gbar_sum += _calc_gbar(pop[ind], d)
    gbar_mean = gbar_sum / N_pop

    if gbar_mean < h_star:
        state = int32(0)
    elif gbar_mean > 1.0 - h_star:
        state = int32(2)
    else:
        state = int32(1)

    last_basin = state if state != int32(1) else int32(-1)
    basin_entry = int64(0)
    transit_entry = int64(0)
    in_transit = False

    fitnesses = np.empty(N_pop, dtype=float64)

    for gen in range(n_gens):
        # --- Mutation ---
        for ind in range(N_pop):
            for loc in range(d):
                if np.random.random() < mu_rate:
                    pop[ind, loc] = int32(1) - pop[ind, loc]

        # --- Fitness evaluation ---
        for ind in range(N_pop):
            gb = _calc_gbar(pop[ind], d)
            nk = _nk_fitness(pop[ind], d, K, neighbors, fi_flat, table_size)
            b1 = np.exp(-2.0 * gb * gb)
            b2 = np.exp(-2.0 * (1.0 - gb) * (1.0 - gb))
            fitnesses[ind] = nk + beta * max(b1, b2)

        # --- Selection (fitness-proportional Wright-Fisher) ---
        w_mean = 0.0
        for ind in range(N_pop):
            w_mean += fitnesses[ind]
        w_mean /= N_pop

        weights = np.empty(N_pop, dtype=float64)
        w_sum = 0.0
        for ind in range(N_pop):
            weights[ind] = np.exp(scoeff * (fitnesses[ind] - w_mean))
            w_sum += weights[ind]
        for ind in range(N_pop):
            weights[ind] /= w_sum

        # Cumulative distribution
        cum = np.empty(N_pop, dtype=float64)
        cum[0] = weights[0]
        for ind in range(1, N_pop):
            cum[ind] = cum[ind - 1] + weights[ind]

        # Sample next generation
        new_pop = np.empty((N_pop, d), dtype=int32)
        for ind in range(N_pop):
            r = np.random.random()
            chosen = 0
            for j in range(N_pop):
                if r <= cum[j]:
                    chosen = j
                    break
            for loc in range(d):
                new_pop[ind, loc] = pop[chosen, loc]

        for ind in range(N_pop):
            for loc in range(d):
                pop[ind, loc] = new_pop[ind, loc]

        # --- Centroid tracking ---
        gbar_sum = 0.0
        for ind in range(N_pop):
            gbar_sum += _calc_gbar(pop[ind], d)
        gbar_mean = gbar_sum / N_pop

        if gbar_mean < h_star:
            state = int32(0)
        elif gbar_mean > 1.0 - h_star:
            state = int32(2)
        else:
            state = int32(1)

        # Transition tracking
        gen64 = int64(gen)
        if not in_transit:
            if state == int32(1):
                in_transit = True
                transit_entry = gen64
            elif state != int32(1):
                if last_basin == int32(-1):
                    last_basin = state
                    basin_entry = gen64
                elif state != last_basin:
                    if n_tr < max_tr:
                        stasis_arr[n_tr] = gen64 - basin_entry
                        transit_arr[n_tr] = int64(1)
                        n_tr += 1
                    last_basin = state
                    basin_entry = gen64
        else:
            if state != int32(1):
                td = gen64 - transit_entry
                if state != last_basin and last_basin >= int32(0):
                    if n_tr < max_tr:
                        stasis_arr[n_tr] = transit_entry - basin_entry
                        transit_arr[n_tr] = td
                        n_tr += 1
                last_basin = state
                basin_entry = gen64
                in_transit = False

    return stasis_arr[:n_tr], transit_arr[:n_tr], n_tr


# ============================================================
# Parameter classes
# ============================================================

@dataclass
class PolyParams:
    """Parameters for polynomial basin shape experiment."""
    d: int; K: int; beta: float; scoeff: float
    n_steps: int; seed: int; h_star: float
    p_shape: float  # -1 = Gaussian, >0 = polynomial exponent


@dataclass
class WFParams:
    """Parameters for Wright-Fisher population experiment."""
    d: int; K: int; beta: float; scoeff: float
    n_steps: int; seed: int; h_star: float
    N_pop: int       # Population size
    mu_rate: float   # Per-locus mutation rate per generation


# ============================================================
# Runner functions (top-level for multiprocessing pickling)
# ============================================================

def run_poly(params: PolyParams) -> Dict[str, Any]:
    """Run one MC simulation with polynomial basin shape."""
    t0 = time.perf_counter()
    d, K = params.d, params.K
    rng = np.random.RandomState(params.seed)

    neighbors = _build_neighbors(d, K) if K > 0 else np.zeros((d, 1), dtype=np.int32)
    table_size = 1 << (K + 1)
    fi_flat = rng.random(d * table_size).astype(np.float64)

    n = params.n_steps
    rand_loci = rng.randint(0, d, size=n).astype(np.int32)
    rand_accept = rng.random(n).astype(np.float64)
    n_flip = max(1, d // 10)
    rand_init = rng.choice(d, size=n_flip, replace=False).astype(np.int32)

    sd, td, n_tr = _mc_core_poly(
        d, K, params.beta, params.scoeff, params.h_star, n,
        params.p_shape,
        neighbors, fi_flat, int32(table_size),
        rand_loci, rand_accept, rand_init
    )
    n_tr = int(n_tr)

    if n_tr > 0:
        ms, mt = float(np.mean(sd)), float(np.mean(td))
        ratio = ms / mt if mt > 0 else 1e10
    else:
        ms, mt, ratio = float(n), 0.0, 1e10

    return {
        'd': d, 'K': K, 'beta': params.beta, 'scoeff': params.scoeff,
        'h_star': params.h_star, 'p_shape': params.p_shape,
        'n_steps': n, 'seed': params.seed,
        'n_transitions': n_tr, 'mean_stasis': ms,
        'mean_transit': mt, 'ratio': min(ratio, 1e10),
        'wall_time': time.perf_counter() - t0,
    }


def run_wf(params: WFParams) -> Dict[str, Any]:
    """Run one Wright-Fisher population simulation.

    RNG design: The NK landscape and initial genotype are generated
    by a numpy RandomState instance seeded with params.seed. A second
    seed for the Numba-internal RNG (used for mutation and selection
    inside the JIT function) is derived from the same RandomState,
    ensuring full determinism without correlation between landscape
    and dynamics.
    """
    t0 = time.perf_counter()
    d, K = params.d, params.K
    rng = np.random.RandomState(params.seed)

    neighbors = _build_neighbors(d, K) if K > 0 else np.zeros((d, 1), dtype=np.int32)
    table_size = 1 << (K + 1)
    fi_flat = rng.random(d * table_size).astype(np.float64)
    n_flip = max(1, d // 10)
    rand_init = rng.choice(d, size=n_flip, replace=False).astype(np.int32)

    # Derive a separate seed for Numba-internal RNG from the same stream
    wf_rng_seed = int(rng.randint(0, 2**31))

    sd, td, n_tr = _mc_core_wf(
        d, K, params.beta, params.scoeff, params.h_star,
        params.n_steps, params.N_pop, params.mu_rate,
        neighbors, fi_flat, int32(table_size), rand_init,
        wf_rng_seed
    )
    n_tr = int(n_tr)

    if n_tr > 0:
        ms, mt = float(np.mean(sd)), float(np.mean(td))
        ratio = ms / mt if mt > 0 else 1e10
    else:
        ms, mt, ratio = float(params.n_steps), 0.0, 1e10

    return {
        'd': d, 'K': K, 'beta': params.beta, 'scoeff': params.scoeff,
        'h_star': params.h_star, 'N_pop': params.N_pop,
        'mu_rate': params.mu_rate, 'Nmu': params.N_pop * params.mu_rate,
        'n_steps': params.n_steps, 'seed': params.seed,
        'n_transitions': n_tr, 'mean_stasis': ms,
        'mean_transit': mt, 'ratio': min(ratio, 1e10),
        'wall_time': time.perf_counter() - t0,
    }
