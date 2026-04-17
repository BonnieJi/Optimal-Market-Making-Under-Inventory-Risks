"""
Step 10: parameter sensitivity — sweep model inputs and summarise Monte Carlo outcomes.

Each sweep re-runs the simulator for many seeds at each parameter value and plots
mean terminal PnL, mean half-spread (quote width), and mean |q_T| (liquidation proxy).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import (
    run_avellaneda_stoikov_path,
    run_fixed_spread_path,
    run_inventory_aware_path,
)
from metrics import PathMetrics, compute_path_metrics, summarize_paths
from plotting import plot_sensitivity_line, plot_sensitivity_twinx
from strategy import as_optimal_half_spread, as_reservation_price


def _default_phys() -> dict:
    return {
        "S0": 100.0,
        "sigma": 1.0,
        "A": 40.0,
        "dt": 0.01,
        "T": 1.0,
        "k": 1.0,
    }


def _mc_paths(
    runner,
    *,
    n_paths: int,
    base_seed: int,
    **kwargs,
):
    paths: list[PathMetrics] = []
    for i in range(n_paths):
        st = runner(seed=base_seed + i, **kwargs)
        paths.append(compute_path_metrics(st))
    return summarize_paths(paths)


def run_parameter_sweeps(
    *,
    n_paths: int = 120,
    base_seed: int = 0,
    results_dir: Path | None = None,
) -> Path:
    """Run 1D sweeps for γ, σ, k, c, α, and horizon T; write PNGs to results_dir."""
    phys = _default_phys()
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent / "results" / "sensitivity"

    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Avellaneda–Stoikov: gamma (risk aversion) ---
    gammas = np.geomspace(0.02, 0.5, num=8)
    mt_pnl, mhs, mq = [], [], []
    theo_delta = []
    tau_mid = float(phys["T"]) * 0.5
    for g in gammas:
        s = _mc_paths(
            run_avellaneda_stoikov_path,
            n_paths=n_paths,
            base_seed=base_seed,
            gamma=float(g),
            **phys,
        )
        mt_pnl.append(s.mean_terminal_pnl)
        mhs.append(s.mean_half_spread)
        mq.append(s.mean_terminal_abs_q)
        theo_delta.append(
            as_optimal_half_spread(
                float(g), phys["sigma"], phys["k"], tau_mid
            )
        )
    gammas_arr = np.asarray(gammas, dtype=float)
    plot_sensitivity_twinx(
        gammas_arr,
        np.asarray(mt_pnl),
        np.asarray(mhs),
        xlabel="γ (risk aversion)",
        left_ylabel="Mean terminal PnL",
        right_ylabel="Mean half-spread (sim)",
        title="AS: higher γ → wider quotes (less aggressive)",
        save_path=results_dir / "sweep_gamma_as_pnl_spread.png",
    )
    plot_sensitivity_line(
        gammas_arr,
        np.asarray(theo_delta),
        xlabel="γ",
        ylabel=f"Theoretical δ at τ=T/2={tau_mid:.2f}",
        title="AS closed-form half-spread vs γ (σ, k fixed)",
        save_path=results_dir / "sweep_gamma_as_theoretical_delta.png",
    )
    plot_sensitivity_twinx(
        gammas_arr,
        np.asarray(mt_pnl),
        np.asarray(mq),
        xlabel="γ (risk aversion)",
        left_ylabel="Mean terminal PnL",
        right_ylabel="Mean |q_T|",
        title="AS: inventory at horizon vs γ",
        save_path=results_dir / "sweep_gamma_as_pnl_terminal_q.png",
    )

    # --- AS: sigma (volatility) ---
    sigmas = np.linspace(0.25, 2.5, num=8)
    mt_pnl, mhs = [], []
    theo_delta_s = []
    for sig in sigmas:
        p = {**phys, "sigma": float(sig)}
        g0 = 0.1
        s = _mc_paths(
            run_avellaneda_stoikov_path,
            n_paths=n_paths,
            base_seed=base_seed,
            gamma=g0,
            **p,
        )
        mt_pnl.append(s.mean_terminal_pnl)
        mhs.append(s.mean_half_spread)
        theo_delta_s.append(as_optimal_half_spread(g0, float(sig), phys["k"], tau_mid))
    sigmas_arr = np.asarray(sigmas, dtype=float)
    plot_sensitivity_twinx(
        sigmas_arr,
        np.asarray(mt_pnl),
        np.asarray(mhs),
        xlabel="σ (volatility)",
        left_ylabel="Mean terminal PnL",
        right_ylabel="Mean half-spread (sim)",
        title="AS: higher σ → wider spreads / more cautious quotes",
        save_path=results_dir / "sweep_sigma_as.png",
    )
    plot_sensitivity_line(
        sigmas_arr,
        np.asarray(theo_delta_s),
        xlabel="σ",
        ylabel=f"Theoretical δ at τ=T/2",
        title="AS theoretical half-spread vs σ (γ=0.1)",
        save_path=results_dir / "sweep_sigma_as_theoretical.png",
    )

    # --- AS: k (arrival sensitivity) ---
    ks = np.geomspace(0.4, 4.0, num=8)
    mt_pnl, mhs = [], []
    phys_no_k = {key: v for key, v in phys.items() if key != "k"}
    for k_val in ks:
        s = _mc_paths(
            run_avellaneda_stoikov_path,
            n_paths=n_paths,
            base_seed=base_seed,
            gamma=0.1,
            k=float(k_val),
            **phys_no_k,
        )
        mt_pnl.append(s.mean_terminal_pnl)
        mhs.append(s.mean_half_spread)
    ks_arr = np.asarray(ks, dtype=float)
    plot_sensitivity_twinx(
        ks_arr,
        np.asarray(mt_pnl),
        np.asarray(mhs),
        xlabel="k (fill decay)",
        left_ylabel="Mean terminal PnL",
        right_ylabel="Mean half-spread (sim)",
        title="AS: k shifts optimal distance vs fill model",
        save_path=results_dir / "sweep_k_as.png",
    )

    # --- Baseline: fixed half-spread c ---
    cs = np.linspace(0.1, 1.2, num=8)
    mt_pnl, mfills = [], []
    for c in cs:
        s = _mc_paths(
            run_fixed_spread_path,
            n_paths=n_paths,
            base_seed=base_seed,
            c=float(c),
            **phys,
        )
        mt_pnl.append(s.mean_terminal_pnl)
        mfills.append(s.mean_fills)
    cs_arr = np.asarray(cs, dtype=float)
    plot_sensitivity_twinx(
        cs_arr,
        np.asarray(mt_pnl),
        np.asarray(mfills),
        xlabel="c (half-spread)",
        left_ylabel="Mean terminal PnL",
        right_ylabel="Mean total fills",
        title="Fixed spread: wide c → fewer fills, different PnL",
        save_path=results_dir / "sweep_c_baseline.png",
    )

    # --- Heuristic: inventory penalty alpha ---
    alphas = np.linspace(0.02, 0.35, num=8)
    mt_pnl, mq = [], []
    c0 = 0.5
    for a in alphas:
        s = _mc_paths(
            run_inventory_aware_path,
            n_paths=n_paths,
            base_seed=base_seed,
            c=c0,
            alpha=float(a),
            **phys,
        )
        mt_pnl.append(s.mean_terminal_pnl)
        mq.append(s.mean_abs_inventory)
    alphas_arr = np.asarray(alphas, dtype=float)
    plot_sensitivity_twinx(
        alphas_arr,
        np.asarray(mt_pnl),
        np.asarray(mq),
        xlabel="α (inventory penalty)",
        left_ylabel="Mean terminal PnL",
        right_ylabel="Mean time-avg |q|",
        title="Heuristic: larger α pushes quotes to flatten inventory",
        save_path=results_dir / "sweep_alpha_heuristic.png",
    )

    # --- AS: horizon T (time to horizon scales τ and liquidation pressure) ---
    Ts = np.linspace(0.25, 2.0, num=7)
    mt_pnl, mtermq = [], []
    r_shift_ref = []
    q_ref = 3.0
    S_ref = phys["S0"]
    for T_val in Ts:
        pT = {**phys, "T": float(T_val)}
        s = _mc_paths(
            run_avellaneda_stoikov_path,
            n_paths=n_paths,
            base_seed=base_seed,
            gamma=0.1,
            **pT,
        )
        mt_pnl.append(s.mean_terminal_pnl)
        mtermq.append(s.mean_terminal_abs_q)
        tau0 = float(T_val)
        r_shift_ref.append(
            abs(
                S_ref
                - as_reservation_price(S_ref, q_ref, 0.1, phys["sigma"], tau0)
            )
        )
    Ts_arr = np.asarray(Ts, dtype=float)
    plot_sensitivity_twinx(
        Ts_arr,
        np.asarray(mt_pnl),
        np.asarray(mtermq),
        xlabel="T (session horizon)",
        left_ylabel="Mean terminal PnL",
        right_ylabel="Mean |q_T|",
        title="AS: longer horizon affects τ-weighted spread and inventory exit",
        save_path=results_dir / "sweep_T_horizon_as.png",
    )
    plot_sensitivity_line(
        Ts_arr,
        np.asarray(r_shift_ref),
        xlabel="T",
        ylabel="|S − r| at t=0 for q=3 (closed-form)",
        title="Reservation-price tilt vs horizon (illustration)",
        save_path=results_dir / "sweep_T_reservation_tilt.png",
    )

    print(f"Sensitivity figures written under: {results_dir.resolve()}")
    return results_dir


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    run_parameter_sweeps(n_paths=n)
