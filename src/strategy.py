from dataclasses import dataclass

import numpy as np

from market import MarketState


@dataclass
class FixedSpreadBaseline:
#fixed half spread c = constent delta

    c: float = 0.5

    def apply(self, st: MarketState) -> None:
        st.bid = st.S - self.c
        st.ask = st.S + self.c


@dataclass
class InventoryAwareStrategy:

    c: float = 0.5
    alpha: float = 0.1

    def apply(self, st: MarketState) -> None:
        r_t = st.S - self.alpha * st.q
        st.bid = r_t - self.c
        st.ask = r_t + self.c


@dataclass
class AvellanedaStoikovStrategy:
    """
    Avellaneda–Stoikov (simplified closed form): quote symmetrically around a
    reservation price with a time- and parameter-dependent half-spread.

        tau = T - t
        r_t = S_t - q_t * gamma * sigma**2 * tau
        delta_t = (1/gamma) * log(1 + gamma/k) + (gamma * sigma**2 / 2) * tau
        bid = r_t - delta_t
        ask = r_t + delta_t

    Here `k` matches the order-flow decay in the simulator fill model (same symbol
    as in the queue intensity lambda ~ exp(-k * distance)).
    """

    gamma: float = 0.1
    sigma: float = 1.0
    T: float = 1.0
    k: float = 1.0

    def apply(self, st: MarketState) -> None:
        tau = max(0.0, float(self.T - st.t))
        sig2 = float(self.sigma**2)
        r_t = float(st.S) - float(st.q) * self.gamma * sig2 * tau
        # Symmetric optimal half-spread (standard reduced form)
        delta_t = (1.0 / self.gamma) * np.log(1.0 + self.gamma / self.k)
        delta_t += 0.5 * self.gamma * sig2 * tau
        st.bid = r_t - delta_t
        st.ask = r_t + delta_t


def as_reservation_price(S: float, q: float, gamma: float, sigma: float, tau: float) -> float:
    """Closed-form reservation price (same sign convention as AvellanedaStoikovStrategy)."""
    return float(S) - float(q) * gamma * float(sigma**2) * float(tau)


def as_optimal_half_spread(gamma: float, sigma: float, k: float, tau: float) -> float:
    """Symmetric optimal half-spread δ_t at time-to-horizon τ (reduced AS formula)."""
    sig2 = float(sigma**2)
    return (1.0 / gamma) * np.log(1.0 + gamma / k) + 0.5 * gamma * sig2 * tau
