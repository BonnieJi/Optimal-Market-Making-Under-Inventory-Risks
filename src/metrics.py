"""
Path-level and Monte Carlo summaries for strategy comparison.

Terminal PnL is mark-to-market: W_t - W_0 from simulated wealth history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def mark_to_market_pnl_series(state) -> np.ndarray:
    w = np.asarray(state.hist_W, dtype=float)
    if w.size == 0:
        return w
    return w - w[0]


def max_drawdown_from_pnl(pnl: np.ndarray) -> float:
    """Largest peak-to-trough drop on the PnL path (<= 0)."""
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        return 0.0
    peak = np.maximum.accumulate(pnl)
    dd = pnl - peak
    return float(np.min(dd))


@dataclass(frozen=True)
class PathMetrics:
    terminal_pnl: float
    max_drawdown: float
    mean_abs_q: float
    var_q: float
    terminal_abs_q: float
    mean_half_spread: float
    fills_buy: int
    fills_sell: int
    cumulative_fills: int


def compute_path_metrics(state) -> PathMetrics:
    pnl = mark_to_market_pnl_series(state)
    q = np.asarray(state.hist_q, dtype=float)
    mean_abs_q = float(np.mean(np.abs(q))) if q.size else 0.0
    var_q = float(np.var(q, ddof=1)) if q.size > 1 else 0.0
    terminal_abs_q = float(np.abs(q[-1])) if q.size else float(abs(state.q))
    bid = np.asarray(state.hist_bid, dtype=float)
    ask = np.asarray(state.hist_ask, dtype=float)
    if bid.size and ask.size and bid.size == ask.size:
        mean_half_spread = float(np.mean(0.5 * (ask - bid)))
    else:
        mean_half_spread = 0.0
    return PathMetrics(
        terminal_pnl=float(pnl[-1]) if pnl.size else 0.0,
        max_drawdown=max_drawdown_from_pnl(pnl),
        mean_abs_q=mean_abs_q,
        var_q=var_q,
        terminal_abs_q=terminal_abs_q,
        mean_half_spread=mean_half_spread,
        fills_buy=int(state.fills_buy),
        fills_sell=int(state.fills_sell),
        cumulative_fills=int(state.cumulative_fills),
    )


@dataclass(frozen=True)
class MonteCarloSummary:
    mean_terminal_pnl: float
    std_terminal_pnl: float
    sharpe_like: float
    mean_max_drawdown: float
    mean_abs_inventory: float
    mean_inventory_var: float
    mean_terminal_abs_q: float
    mean_half_spread: float
    mean_fills: float
    mean_fills_buy: float
    mean_fills_sell: float


def summarize_paths(paths: list[PathMetrics]) -> MonteCarloSummary:
    if not paths:
        raise ValueError("paths must be non-empty")
    term = np.array([p.terminal_pnl for p in paths], dtype=float)
    dd = np.array([p.max_drawdown for p in paths], dtype=float)
    maq = np.array([p.mean_abs_q for p in paths], dtype=float)
    vq = np.array([p.var_q for p in paths], dtype=float)
    term_q = np.array([p.terminal_abs_q for p in paths], dtype=float)
    mhs = np.array([p.mean_half_spread for p in paths], dtype=float)
    fills = np.array([p.cumulative_fills for p in paths], dtype=float)
    fb = np.array([p.fills_buy for p in paths], dtype=float)
    fs = np.array([p.fills_sell for p in paths], dtype=float)

    std_t = float(np.std(term, ddof=1)) if term.size > 1 else 0.0
    mean_t = float(np.mean(term))
    sharpe = mean_t / std_t if std_t > 1e-12 else float("nan")

    return MonteCarloSummary(
        mean_terminal_pnl=mean_t,
        std_terminal_pnl=std_t,
        sharpe_like=sharpe,
        mean_max_drawdown=float(np.mean(dd)),
        mean_abs_inventory=float(np.mean(maq)),
        mean_inventory_var=float(np.mean(vq)),
        mean_terminal_abs_q=float(np.mean(term_q)),
        mean_half_spread=float(np.mean(mhs)),
        mean_fills=float(np.mean(fills)),
        mean_fills_buy=float(np.mean(fb)),
        mean_fills_sell=float(np.mean(fs)),
    )
