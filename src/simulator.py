from __future__ import annotations

import numpy as np
from market import MarketState
import signals as signal_ops
from strategy import (
    AvellanedaStoikovOFIStrategy,
    AvellanedaStoikovStrategy,
    FixedSpreadBaseline,
    InventoryAwareStrategy,
)


class MarketSimulator:
    def __init__(
        self,
        S0: float = 100.0,  # init mid price
        sigma: float = 1.0,  # volatility per square step
        A: float = 1.0,  # Avellaneda-Stoikov lamda poisson arrival rate per step
        delta: float = 0.5,  # half-spread c (fixed-spread baseline)
        dt: float = 0.01,  # time steps
        T: float = 1.0,  # horizon trading session
        k: float = 1.0,  # order arrival elasticity
        seed: int | None = None,
        strategy: FixedSpreadBaseline
        | InventoryAwareStrategy
        | AvellanedaStoikovStrategy
        | AvellanedaStoikovOFIStrategy
        | None = None,
        # realism bonuses (all optional; defaults preserve previous behavior)
        adverse_selection_prob: float = 0.0,
        adverse_selection_impact: float = 0.0,
        latency_steps: int = 0,
        sigma_regimes: tuple[float, ...] | None = None,
        regime_transition: np.ndarray | None = None,
        hawkes_decay: float = 0.0,
        hawkes_jump: float = 0.0,
        fee_per_unit: float = 0.0,
        slippage_per_unit: float = 0.0,
        q_max: int | None = None,
        q_min: int | None = None,
        lob_evolve_vol: float = 0.0,
        ofi_window_len: int = 50,
    ):
        self.S0 = S0
        self.sigma = sigma
        self.A = A
        self.strategy = strategy if strategy is not None else FixedSpreadBaseline(c=delta)
        # reference half-spread for baselines that use fixed c; AS has no single c
        self.delta = getattr(self.strategy, "c", delta)
        self.dt = dt
        self.T = T
        self.k = k
        self.n_steps = int(T / dt)
        self.adverse_selection_prob = float(adverse_selection_prob)
        self.adverse_selection_impact = float(adverse_selection_impact)
        self.latency_steps = int(max(0, latency_steps))
        self.sigma_regimes = tuple(float(x) for x in sigma_regimes) if sigma_regimes else None
        self.regime_transition = regime_transition
        self.hawkes_decay = float(hawkes_decay)
        self.hawkes_jump = float(hawkes_jump)
        self.fee_per_unit = float(max(0.0, fee_per_unit))
        self.slippage_per_unit = float(max(0.0, slippage_per_unit))
        self.q_max = q_max
        self.q_min = q_min
        self.lob_evolve_vol = float(max(0.0, lob_evolve_vol))
        self.ofi_window_len = int(max(1, ofi_window_len))

        self.rng = np.random.default_rng(seed)
        self.state: MarketState | None = None
        self.current_sigma = float(self.sigma)
        self.current_regime = 0
        self.hawkes_excitation = 0.0

    def reset(self) -> MarketState:
        st = MarketState(S=self.S0)
        self.current_regime = 0
        self.current_sigma = (
            float(self.sigma_regimes[0]) if self.sigma_regimes else float(self.sigma)
        )
        self.hawkes_excitation = 0.0

        # initial quotes from strategy (fixed spread, heuristic, or AS)
        self.strategy.apply(st)

        self.state = st
        # record initial history
        self._record_history()
        return st

    def _observed_mid_and_time(self) -> tuple[float, float]:
        """Price/time seen by strategy after quote latency."""
        st = self.state
        assert st is not None
        if self.latency_steps <= 0:
            return st.S, st.t
        idx = max(0, len(st.hist_S) - 1 - self.latency_steps)
        return float(st.hist_S[idx]), float(st.hist_t[idx])

    def _update_volatility_regime(self) -> None:
        """Optional Markov switching volatility regimes."""
        if not self.sigma_regimes:
            self.current_sigma = float(self.sigma)
            return

        n = len(self.sigma_regimes)
        if self.regime_transition is None:
            # simple sticky chain: high persistence with equal leakage to other states
            stay = 0.92
            move = (1.0 - stay) / (n - 1) if n > 1 else 0.0
            probs = np.full((n, n), move, dtype=float)
            np.fill_diagonal(probs, stay)
        else:
            probs = np.asarray(self.regime_transition, dtype=float)
            if probs.shape != (n, n):
                raise ValueError("regime_transition must match sigma_regimes shape")
        row = probs[self.current_regime]
        self.current_regime = int(self.rng.choice(n, p=row))
        self.current_sigma = float(self.sigma_regimes[self.current_regime])

    def _update_mid_price(self) -> None:
        #Arithmetic Brownian motion: dS = sigma dW.
        st = self.state
        assert st is not None

        epsilon = self.rng.normal()  # generate random price shock
        st.S = st.S + self.current_sigma * np.sqrt(self.dt) * epsilon

        # floor - avoid negative prices in arithmetic BM
        st.S = max(1e-4, st.S)

        if self.lob_evolve_vol > 0.0:
            signal_ops.evolve_lob_depths(st, self.rng, self.dt, self.lob_evolve_vol)

    def _shadow_state(self, obs_S: float, obs_t: float) -> MarketState:
        st = self.state
        assert st is not None
        shadow = MarketState(t=obs_t, S=obs_S, q=st.q, X=st.X, bid=st.bid, ask=st.ask)
        shadow.ofi_window = list(st.ofi_window)
        shadow.bid_depth = st.bid_depth
        shadow.ask_depth = st.ask_depth
        shadow.last_alpha = st.last_alpha
        shadow.fills_buy = st.fills_buy
        shadow.fills_sell = st.fills_sell
        return shadow

    def _enforce_inventory_limits(self, st: MarketState) -> None:
        """Hard caps: stop quoting the side that would increase breach."""
        if self.q_max is not None and st.q >= self.q_max:
            st.bid = st.S - 1e3
        if self.q_min is not None and st.q <= self.q_min:
            st.ask = st.S + 1e3

    def _update_quotes(self) -> None:
        # strategy posts bid/ask (fixed spread, reservation price, or AS)
        st = self.state
        assert st is not None

        obs_S, obs_t = self._observed_mid_and_time()
        # Strategy computes quotes from stale view, but fills still use current st.S.
        shadow = self._shadow_state(obs_S, obs_t)
        self.strategy.apply(shadow)
        st.bid = shadow.bid
        st.ask = shadow.ask
        st.last_alpha = shadow.last_alpha
        self._enforce_inventory_limits(st)

    def _fill_probabilities(self) -> tuple[float, float]:

        st = self.state
        assert st is not None

        delta_b = st.S - st.bid
        delta_a = st.ask - st.S

        lambda_b = self.A * np.exp(-self.k * delta_b)
        lambda_a = self.A * np.exp(-self.k * delta_a)
        if self.hawkes_excitation > 0.0:
            boost = 1.0 + self.hawkes_excitation
            lambda_b *= boost
            lambda_a *= boost

        # poisson arrival approximation (Bernoulli per step)
        p_bid = min(lambda_b * self.dt, 1.0)
        p_ask = min(lambda_a * self.dt, 1.0)

        return p_bid, p_ask

    def _process_fills(self) -> None:
        st = self.state
        assert st is not None

        p_bid, p_ask = self._fill_probabilities()

        bid_fill = self.rng.random() < p_bid
        ask_fill = self.rng.random() < p_ask

        # bid hit: we buy — inventory up, cash down by effective price (slip + fee)
        if bid_fill:  # buy
            st.q += 1
            buy_px = st.bid + self.slippage_per_unit
            st.X -= buy_px + self.fee_per_unit
            st.fills_buy += 1

        # ask hit: we sell — inventory down, cash up by effective price
        if ask_fill:  # sell
            st.q -= 1
            sell_px = st.ask - self.slippage_per_unit
            st.X += sell_px - self.fee_per_unit
            st.fills_sell += 1

        if bid_fill or ask_fill:
            signal_ops.record_aggressive_trade(
                st, ask_hit=ask_fill, bid_hit=bid_fill, max_len=self.ofi_window_len
            )
            signal_ops.apply_fill_to_lob(st, ask_hit=ask_fill, bid_hit=bid_fill)

        # Hawkes-lite clustered arrivals: recent fills boost near-future intensity.
        if self.hawkes_decay > 0.0 and self.hawkes_excitation > 0.0:
            self.hawkes_excitation *= np.exp(-self.hawkes_decay * self.dt)
        if self.hawkes_jump > 0.0:
            self.hawkes_excitation += self.hawkes_jump * float(bid_fill or ask_fill)

        # Adverse selection drift: price moves against maker after being picked off.
        if self.adverse_selection_prob > 0.0 and self.adverse_selection_impact > 0.0:
            adverse_move = 0.0
            if bid_fill and self.rng.random() < self.adverse_selection_prob:
                adverse_move -= self.adverse_selection_impact
            if ask_fill and self.rng.random() < self.adverse_selection_prob:
                adverse_move += self.adverse_selection_impact
            if adverse_move != 0.0:
                st.S = max(1e-4, st.S + adverse_move)

    def _record_history(self) -> None:
        st = self.state
        assert st is not None

        st.hist_t.append(st.t)
        st.hist_S.append(st.S)
        st.hist_bid.append(st.bid)
        st.hist_ask.append(st.ask)
        st.hist_q.append(st.q)
        st.hist_X.append(st.X)
        st.hist_W.append(st.wealth)
        st.hist_alpha.append(st.last_alpha)

    def step(self) -> MarketState:
        st = self.state
        assert st is not None

        # advance time
        st.t = round(st.t + self.dt, 10)

        self._update_volatility_regime()
        self._update_mid_price()
        self._update_quotes()
        self._process_fills()
        # record history
        self._record_history()

        return st

    def run(self) -> MarketState:
        self.reset()
        for _ in range(self.n_steps):
            self.step()
        assert self.state is not None
        return self.state


# --- Quick demo ---
if __name__ == "__main__":
    sim = MarketSimulator(
        S0=100.0,
        sigma=1.0,
        A=1.0,
        delta=0.5,
        dt=0.01,
        T=1.0,
        k=1.0,
        seed=42,
    )

    final = sim.run()

    print(f"Final mid-price : {final.S:.4f}")
    print(f"Inventory q_T   : {final.q}")
    print(f"Cash X_T        : {final.X:.4f}")
    print(f"Wealth W_T      : {final.wealth:.4f}")
    print(f"Buy fills       : {final.fills_buy}")
    print(f"Sell fills      : {final.fills_sell}")
    print(f"Cumulative fills: {final.cumulative_fills}")
