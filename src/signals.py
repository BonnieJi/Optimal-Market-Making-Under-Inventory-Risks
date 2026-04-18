"""
Order-flow style alpha for quote skew.

OFI (order flow imbalance, trade-based):
    aggressive buy volume − aggressive sell volume

We proxy this from recent *taker* activity against our quotes:
    ask filled  → aggressive buy  → +1
    bid filled  → aggressive sell → −1

LOB imbalance (depth-based):
    (bid_depth − ask_depth) / (bid_depth + ask_depth)

Combined alpha ∈ [−1, 1] via tanh for stable skew into the reservation price.
"""

from __future__ import annotations

import math

from market import MarketState


def lob_imbalance(st: MarketState) -> float:
    d = float(st.bid_depth + st.ask_depth)
    if d <= 1e-9:
        return 0.0
    return float(st.bid_depth - st.ask_depth) / d


def ofi_sum_window(st: MarketState) -> float:
    """Sum of signed aggressive flow over the retained window."""
    return float(sum(st.ofi_window))


def compute_alpha_signal(
    st: MarketState,
    *,
    w_ofi: float = 0.55,
    w_lob: float = 0.45,
    ofi_scale: float = 8.0,
) -> float:
    """
    Bounded predictive signal for skewing quotes.

    w_ofi, w_lob blend trade OFI vs book imbalance; ofi_scale normalizes raw OFI sum.
    """
    raw_ofi = ofi_sum_window(st)
    z_ofi = raw_ofi / max(1e-6, ofi_scale)
    imb = lob_imbalance(st)
    z = w_ofi * z_ofi + w_lob * imb
    return float(math.tanh(z))


def evolve_lob_depths(st: MarketState, rng: np.random.Generator, dt: float, vol: float) -> None:
    """Small mean-reverting noise on synthetic depth (stylized LOB)."""
    if vol <= 0.0:
        return
    shock = vol * math.sqrt(max(dt, 1e-12))
    st.bid_depth = max(1.0, st.bid_depth + float(rng.normal(0.0, shock)))
    st.ask_depth = max(1.0, st.ask_depth + float(rng.normal(0.0, shock)))


def record_aggressive_trade(st: MarketState, *, ask_hit: bool, bid_hit: bool, max_len: int) -> None:
    """Update OFI window from taker aggression against our quotes."""
    if ask_hit:
        st.ofi_window.append(1)
    if bid_hit:
        st.ofi_window.append(-1)
    while len(st.ofi_window) > max_len:
        st.ofi_window.pop(0)


def apply_fill_to_lob(st: MarketState, *, ask_hit: bool, bid_hit: bool) -> None:
    """Liquidity consumed / replenished at our touch (crude book dynamics)."""
    if bid_hit:
        st.bid_depth = max(1.0, st.bid_depth - 2.0)
        st.ask_depth += 0.5
    if ask_hit:
        st.ask_depth = max(1.0, st.ask_depth - 2.0)
        st.bid_depth += 0.5
