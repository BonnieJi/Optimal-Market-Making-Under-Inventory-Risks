"""
 fixed-spread baseline — one path + sanity checks.

Mark-to-market PnL over time: W_t - W_0 (wealth from MarketState).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from metrics import (
    PathMetrics,
    compute_path_metrics,
    mark_to_market_pnl_series,
    summarize_paths,
)
from plotting import plot_drawdown_histogram, plot_sample_path, plot_terminal_pnl_histogram
from simulator import MarketSimulator
from strategy import (
    AvellanedaStoikovOFIStrategy,
    AvellanedaStoikovStrategy,
    FixedSpreadBaseline,
    InventoryAwareStrategy,
)


def run_fixed_spread_path(
    *,
    c: float = 0.5,
    seed: int = 42,
    S0: float = 100.0,
    sigma: float = 1.0,
    A: float = 1.0,
    dt: float = 0.01,
    T: float = 1.0,
    k: float = 1.0,
    **sim_kwargs,
):
    """Single simulation path using the control strategy with half-spread c."""
    sim = MarketSimulator(
        S0=S0,
        sigma=sigma,
        A=A,
        dt=dt,
        T=T,
        k=k,
        seed=seed,
        strategy=FixedSpreadBaseline(c=c),
        **sim_kwargs,
    )
    return sim.run()


def run_inventory_aware_path(
    *,
    c: float = 0.5,
    alpha: float = 0.1,
    seed: int = 42,
    S0: float = 100.0,
    sigma: float = 1.0,
    A: float = 1.0,
    dt: float = 0.01,
    T: float = 1.0,
    k: float = 1.0,
    **sim_kwargs,
):
    """Single path with inventory-aware quotes via reservation price."""
    sim = MarketSimulator(
        S0=S0,
        sigma=sigma,
        A=A,
        dt=dt,
        T=T,
        k=k,
        seed=seed,
        strategy=InventoryAwareStrategy(c=c, alpha=alpha),
        **sim_kwargs,
    )
    return sim.run()


def run_avellaneda_stoikov_path(
    *,
    gamma: float = 0.1,
    seed: int = 42,
    S0: float = 100.0,
    sigma: float = 1.0,
    A: float = 1.0,
    dt: float = 0.01,
    T: float = 1.0,
    k: float = 1.0,
    **sim_kwargs,
):
    """Single path with Avellaneda–Stoikov reservation price and optimal half-spread."""
    sim = MarketSimulator(
        S0=S0,
        sigma=sigma,
        A=A,
        dt=dt,
        T=T,
        k=k,
        seed=seed,
        strategy=AvellanedaStoikovStrategy(gamma=gamma, sigma=sigma, T=T, k=k),
        **sim_kwargs,
    )
    return sim.run()


def run_avellaneda_ofi_path(
    *,
    gamma: float = 0.1,
    skew_scale: float = 0.12,
    seed: int = 42,
    S0: float = 100.0,
    sigma: float = 1.0,
    A: float = 1.0,
    dt: float = 0.01,
    T: float = 1.0,
    k: float = 1.0,
    lob_evolve_vol: float = 4.0,
    fee_per_unit: float = 0.0,
    slippage_per_unit: float = 0.0,
    q_max: int | None = None,
    q_min: int | None = None,
    latency_steps: int = 0,
    **sim_kwargs,
):
    """
    AS + OFI/LOB alpha skew on reservation price, with optional frictions.

    Set fee_per_unit, slippage_per_unit, q_max / q_min, latency_steps for realism.
    """
    sim = MarketSimulator(
        S0=S0,
        sigma=sigma,
        A=A,
        dt=dt,
        T=T,
        k=k,
        seed=seed,
        strategy=AvellanedaStoikovOFIStrategy(
            gamma=gamma, sigma=sigma, T=T, k=k, skew_scale=skew_scale
        ),
        lob_evolve_vol=lob_evolve_vol,
        fee_per_unit=fee_per_unit,
        slippage_per_unit=slippage_per_unit,
        q_max=q_max,
        q_min=q_min,
        latency_steps=latency_steps,
        **sim_kwargs,
    )
    return sim.run()


def verify_fixed_spread_baseline(
    *,
    A: float = 40.0,
    dt: float = 0.01,
    T: float = 1.0,
) -> None:
    """
    Run one path and check:
      - PnL (W_t - W_0) is finite and moves when trading occurs
      - inventory time series is not flat when there are fills
      - tighter c (more aggressive) yields more fills than wider c at same seed

    Default A is raised so Bernoulli arrivals produce fills over the horizon (A=1
    can yield zero fills in short runs).
    """
    seed = 42
    c_mid = 0.5
    final = run_fixed_spread_path(c=c_mid, seed=seed, A=A, dt=dt, T=T)
    pnl = mark_to_market_pnl_series(final)
    q = np.asarray(final.hist_q, dtype=int)

    assert np.all(np.isfinite(pnl)), "PnL path should be finite"
    assert pnl.shape[0] == len(final.hist_t), "history should align with time steps"

    if final.cumulative_fills > 0:
        assert np.nanstd(pnl) > 0 or abs(pnl[-1]) > 1e-9, (
            "with fills, mark-to-market PnL should vary over time"
        )
        rq = float(np.max(q) - np.min(q))
        assert rq > 0, "with fills, inventory should wander (non-constant q)"

    c_tight, c_wide = 0.15, 1.25
    tight = run_fixed_spread_path(c=c_tight, seed=seed, A=A, dt=dt, T=T)
    wide = run_fixed_spread_path(c=c_wide, seed=seed, A=A, dt=dt, T=T)
    assert tight.cumulative_fills >= wide.cumulative_fills, (
        "tighter half-spread should not reduce fills vs wider (same rng stream order)"
    )


def print_path_summary(label: str, state, c: float) -> None:
    pnl = mark_to_market_pnl_series(state)
    q = np.asarray(state.hist_q, dtype=int)
    print(f"\n=== {label} ===")
    print(f"half-spread c     : {c:.4f}")
    print(f"final W           : {state.wealth:.4f}")
    print(f"PnL W_T - W_0     : {pnl[-1]:.4f}")
    print(f"range PnL         : [{pnl.min():.4f}, {pnl.max():.4f}]")
    print(f"inventory range q : [{q.min()}, {q.max()}]  std={float(np.std(q)):.4f}")
    print(f"fills (buy/sell/total): {state.fills_buy}/{state.fills_sell}/{state.cumulative_fills}")


def _print_mc_summary(name: str, s) -> None:
    print(f"\n=== {name} ===")
    print(f"  mean terminal PnL      : {s.mean_terminal_pnl:.6f}")
    print(f"  std terminal PnL       : {s.std_terminal_pnl:.6f}")
    print(f"  Sharpe-like (mean/std) : {s.sharpe_like:.6f}")
    print(f"  mean max drawdown      : {s.mean_max_drawdown:.6f}")
    print(f"  mean |q| (time avg)    : {s.mean_abs_inventory:.6f}")
    print(f"  mean |q_T| (terminal)  : {s.mean_terminal_abs_q:.6f}")
    print(f"  mean half-spread (path): {s.mean_half_spread:.6f}")
    print(f"  mean Var(q) along path : {s.mean_inventory_var:.6f}")
    print(f"  mean fills (total)     : {s.mean_fills:.4f}")
    print(f"  mean fills buy / sell  : {s.mean_fills_buy:.4f} / {s.mean_fills_sell:.4f}")


def compare_baseline_vs_heuristic(
    *,
    n_paths: int = 100,
    base_seed: int = 0,
    c: float = 0.5,
    alpha: float = 0.2,
    S0: float = 100.0,
    sigma: float = 1.0,
    A: float = 40.0,
    dt: float = 0.01,
    T: float = 1.0,
    k: float = 1.0,
    results_dir: Path | None = None,
    sample_seed: int = 42,
    **sim_kwargs,
) -> tuple[list[PathMetrics], list[PathMetrics]]:
    """
    Paired Monte Carlo: same seed per path for baseline vs inventory-aware strategy.

    Saves figures under results_dir when provided (default: ./results next to this file).
    """
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent / "results"

    baseline_paths: list[PathMetrics] = []
    heuristic_paths: list[PathMetrics] = []
    sample_baseline = None
    sample_heuristic = None

    for i in range(n_paths):
        seed = base_seed + i
        b_state = run_fixed_spread_path(
            c=c, seed=seed, S0=S0, sigma=sigma, A=A, dt=dt, T=T, k=k, **sim_kwargs
        )
        h_state = run_inventory_aware_path(
            c=c,
            alpha=alpha,
            seed=seed,
            S0=S0,
            sigma=sigma,
            A=A,
            dt=dt,
            T=T,
            k=k,
            **sim_kwargs,
        )
        baseline_paths.append(compute_path_metrics(b_state))
        heuristic_paths.append(compute_path_metrics(h_state))
        if seed == sample_seed:
            sample_baseline = b_state
            sample_heuristic = h_state

    if sample_baseline is None:
        sample_baseline = run_fixed_spread_path(
            c=c, seed=sample_seed, S0=S0, sigma=sigma, A=A, dt=dt, T=T, k=k, **sim_kwargs
        )
        sample_heuristic = run_inventory_aware_path(
            c=c,
            alpha=alpha,
            seed=sample_seed,
            S0=S0,
            sigma=sigma,
            A=A,
            dt=dt,
            T=T,
            k=k,
            **sim_kwargs,
        )

    sb = summarize_paths(baseline_paths)
    sh = summarize_paths(heuristic_paths)
    _print_mc_summary(f"fixed-spread baseline (n={n_paths})", sb)
    _print_mc_summary(f"inventory-aware α={alpha} (n={n_paths})", sh)

    term_b = np.array([p.terminal_pnl for p in baseline_paths], dtype=float)
    term_h = np.array([p.terminal_pnl for p in heuristic_paths], dtype=float)
    dd_b = np.array([p.max_drawdown for p in baseline_paths], dtype=float)
    dd_h = np.array([p.max_drawdown for p in heuristic_paths], dtype=float)

    plot_sample_path(
        sample_baseline,
        f"Sample path — baseline (seed={sample_seed})",
        save_path=results_dir / "mc_sample_path_baseline.png",
    )
    plot_sample_path(
        sample_heuristic,
        f"Sample path — inventory-aware α={alpha} (seed={sample_seed})",
        save_path=results_dir / "mc_sample_path_heuristic.png",
    )
    plot_terminal_pnl_histogram(
        term_b,
        term_h,
        save_path=results_dir / "mc_terminal_pnl_hist.png",
    )
    plot_drawdown_histogram(
        dd_b,
        dd_h,
        save_path=results_dir / "mc_max_drawdown_hist.png",
    )

    print(f"\nFigures written to: {results_dir.resolve()}")
    return baseline_paths, heuristic_paths


def compare_baseline_vs_avellaneda_stoikov(
    *,
    n_paths: int = 1000,
    base_seed: int = 0,
    c: float = 0.5,
    gamma: float = 0.1,
    S0: float = 100.0,
    sigma: float = 1.0,
    A: float = 40.0,
    dt: float = 0.01,
    T: float = 1.0,
    k: float = 1.0,
    results_dir: Path | None = None,
    sample_seed: int = 42,
) -> tuple[list[PathMetrics], list[PathMetrics]]:
    """
    Paired Monte Carlo: fixed-spread baseline vs Avellaneda–Stoikov (same seed per path).

    Writes figures with `_as` in the filename so they do not overwrite the heuristic run.
    """
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent / "results"

    baseline_paths: list[PathMetrics] = []
    as_paths: list[PathMetrics] = []
    sample_baseline = None
    sample_as = None

    for i in range(n_paths):
        seed = base_seed + i
        b_state = run_fixed_spread_path(
            c=c, seed=seed, S0=S0, sigma=sigma, A=A, dt=dt, T=T, k=k
        )
        a_state = run_avellaneda_stoikov_path(
            gamma=gamma, seed=seed, S0=S0, sigma=sigma, A=A, dt=dt, T=T, k=k
        )
        baseline_paths.append(compute_path_metrics(b_state))
        as_paths.append(compute_path_metrics(a_state))
        if seed == sample_seed:
            sample_baseline = b_state
            sample_as = a_state

    if sample_baseline is None:
        sample_baseline = run_fixed_spread_path(
            c=c, seed=sample_seed, S0=S0, sigma=sigma, A=A, dt=dt, T=T, k=k
        )
        sample_as = run_avellaneda_stoikov_path(
            gamma=gamma, seed=sample_seed, S0=S0, sigma=sigma, A=A, dt=dt, T=T, k=k
        )

    sb = summarize_paths(baseline_paths)
    sas = summarize_paths(as_paths)
    _print_mc_summary(f"fixed-spread baseline (n={n_paths})", sb)
    _print_mc_summary(f"Avellaneda–Stoikov γ={gamma} (n={n_paths})", sas)

    term_b = np.array([p.terminal_pnl for p in baseline_paths], dtype=float)
    term_a = np.array([p.terminal_pnl for p in as_paths], dtype=float)
    dd_b = np.array([p.max_drawdown for p in baseline_paths], dtype=float)
    dd_a = np.array([p.max_drawdown for p in as_paths], dtype=float)

    plot_sample_path(
        sample_baseline,
        f"Sample path — baseline (seed={sample_seed})",
        save_path=results_dir / "mc_sample_path_baseline_as.png",
    )
    plot_sample_path(
        sample_as,
        f"Sample path — Avellaneda–Stoikov γ={gamma} (seed={sample_seed})",
        save_path=results_dir / "mc_sample_path_avellaneda.png",
    )
    plot_terminal_pnl_histogram(
        term_b,
        term_a,
        labels=("baseline", "Avellaneda–Stoikov"),
        title="Terminal PnL (W_T − W_0) — baseline vs AS",
        save_path=results_dir / "mc_terminal_pnl_hist_as.png",
    )
    plot_drawdown_histogram(
        dd_b,
        dd_a,
        labels=("baseline", "Avellaneda–Stoikov"),
        title="Max drawdown per path — baseline vs AS",
        save_path=results_dir / "mc_max_drawdown_hist_as.png",
    )

    print(f"\nFigures written to: {results_dir.resolve()}")
    return baseline_paths, as_paths


def run_phase4_realism(
    *,
    n_paths: int = 500,
    c: float = 0.5,
    alpha: float = 0.2,
    S0: float = 100.0,
    A: float = 40.0,
    dt: float = 0.01,
    T: float = 1.0,
    k: float = 1.0,
) -> None:
    """Phase 4 bonus runs: adverse selection, latency, regimes, and clustered arrivals."""
    print("\n--- Step 11: adverse selection ---")
    compare_baseline_vs_heuristic(
        n_paths=n_paths,
        c=c,
        alpha=alpha,
        S0=S0,
        sigma=1.0,
        A=A,
        dt=dt,
        T=T,
        k=k,
        adverse_selection_prob=0.6,
        adverse_selection_impact=0.03,
    )

    print("\n--- Step 12: latency (1-step) ---")
    compare_baseline_vs_heuristic(
        n_paths=n_paths,
        c=c,
        alpha=alpha,
        S0=S0,
        sigma=1.0,
        A=A,
        dt=dt,
        T=T,
        k=k,
        latency_steps=1,
    )
    print("\n--- Step 12: latency (2-step) ---")
    compare_baseline_vs_heuristic(
        n_paths=n_paths,
        c=c,
        alpha=alpha,
        S0=S0,
        sigma=1.0,
        A=A,
        dt=dt,
        T=T,
        k=k,
        latency_steps=2,
    )

    print("\n--- Step 13: volatility regimes (Markov 3-state) ---")
    compare_baseline_vs_heuristic(
        n_paths=n_paths,
        c=c,
        alpha=alpha,
        S0=S0,
        sigma=1.0,
        A=A,
        dt=dt,
        T=T,
        k=k,
        sigma_regimes=(0.6, 1.0, 1.8),
    )

    print("\n--- Step 14: clustered arrivals (Hawkes-lite) ---")
    compare_baseline_vs_heuristic(
        n_paths=n_paths,
        c=c,
        alpha=alpha,
        S0=S0,
        sigma=1.0,
        A=A,
        dt=dt,
        T=T,
        k=k,
        hawkes_decay=4.0,
        hawkes_jump=0.25,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mc":
        rest = sys.argv[2:]
        n_paths = 500
        for t in rest:
            if t.isdigit():
                n_paths = int(t)
        if "as" in rest:
            compare_baseline_vs_avellaneda_stoikov(n_paths=n_paths)
        else:
            compare_baseline_vs_heuristic(n_paths=n_paths)
    elif len(sys.argv) > 1 and sys.argv[1] == "sweep":
        from sensitivity import run_parameter_sweeps

        rest = sys.argv[2:]
        n_paths = 120
        for t in rest:
            if t.isdigit():
                n_paths = int(t)
        run_parameter_sweeps(n_paths=n_paths)
    elif len(sys.argv) > 1 and sys.argv[1] == "phase4":
        rest = sys.argv[2:]
        n_paths = 500
        for t in rest:
            if t.isdigit():
                n_paths = int(t)
        run_phase4_realism(n_paths=n_paths)
    else:
        verify_fixed_spread_baseline()
        print("verification passed.")

        demo_A = 40.0
        final = run_fixed_spread_path(c=0.5, seed=42, A=demo_A)
        print_path_summary("one path, c=0.5, seed=42", final, c=0.5)

        tight = run_fixed_spread_path(c=0.15, seed=42, A=demo_A)
        wide = run_fixed_spread_path(c=1.25, seed=42, A=demo_A)
        print_path_summary("tight c=0.15 (same seed)", tight, c=0.15)
        print_path_summary("wide  c=1.25 (same seed)", wide, c=1.25)

        inv = run_inventory_aware_path(c=0.5, alpha=0.2, seed=42, A=demo_A)
        print_path_summary("inventory-aware c=0.5, alpha=0.2", inv, c=0.5)

        aso = run_avellaneda_stoikov_path(gamma=0.1, seed=42, A=demo_A)
        pnl_as = mark_to_market_pnl_series(aso)
        q_as = np.asarray(aso.hist_q, dtype=int)
        print(f"\n=== Avellaneda–Stoikov (γ=0.1, seed=42) ===")
        print(f"final W              : {aso.wealth:.4f}")
        print(f"PnL W_T − W_0        : {float(pnl_as[-1]):.4f}")
        print(f"inventory range q    : [{q_as.min()}, {q_as.max()}]")
        print(
            f"fills (buy/sell/total): {aso.fills_buy}/{aso.fills_sell}/{aso.cumulative_fills}"
        )

        ofi = run_avellaneda_ofi_path(
            gamma=0.1,
            seed=42,
            A=demo_A,
            fee_per_unit=0.01,
            slippage_per_unit=0.005,
            q_max=12,
            q_min=-12,
            latency_steps=1,
        )
        pnl_ofi = mark_to_market_pnl_series(ofi)
        a_hist = np.asarray(ofi.hist_alpha, dtype=float)
        print(f"\n=== AS + OFI (fees, slip, |q|≤12, 1-step latency) ===")
        print(f"final W              : {ofi.wealth:.4f}")
        print(f"PnL W_T − W_0        : {float(pnl_ofi[-1]):.4f}")
        print(f"mean |alpha| (path)  : {float(np.mean(np.abs(a_hist))):.4f}")
        print(
            f"fills (buy/sell/total): {ofi.fills_buy}/{ofi.fills_sell}/{ofi.cumulative_fills}"
        )
