"""
 fixed-spread baseline — one path + sanity checks.

Mark-to-market PnL over time: W_t - W_0 (wealth from MarketState).
"""

import numpy as np

from simulator import MarketSimulator
from strategy import FixedSpreadBaseline, InventoryAwareStrategy


def mark_to_market_pnl_series(state) -> np.ndarray:
    w = np.asarray(state.hist_W, dtype=float)
    if w.size == 0:
        return w
    return w - w[0]


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


if __name__ == "__main__":
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
