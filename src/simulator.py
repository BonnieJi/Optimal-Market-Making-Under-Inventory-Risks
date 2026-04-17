from __future__ import annotations

import numpy as np
from market import MarketState
from strategy import AvellanedaStoikovStrategy, FixedSpreadBaseline, InventoryAwareStrategy


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
        strategy: FixedSpreadBaseline | InventoryAwareStrategy | AvellanedaStoikovStrategy | None = None,
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

        self.rng = np.random.default_rng(seed)
        self.state: MarketState | None = None

    def reset(self) -> MarketState:
        st = MarketState(S=self.S0)

        # initial quotes from strategy (fixed spread, heuristic, or AS)
        self.strategy.apply(st)

        self.state = st
        # record initial history
        self._record_history()
        return st

    def _update_mid_price(self) -> None:
        #Arithmetic Brownian motion: dS = sigma dW.
        st = self.state
        assert st is not None

        epsilon = self.rng.normal()  # generate random price shock
        st.S = st.S + self.sigma * np.sqrt(self.dt) * epsilon

        # floor - avoid negative prices in arithmetic BM
        st.S = max(1e-4, st.S)

    def _update_quotes(self) -> None:
        # strategy posts bid/ask (fixed spread, reservation price, or AS)
        st = self.state
        assert st is not None

        self.strategy.apply(st)

    def _fill_probabilities(self) -> tuple[float, float]:

        st = self.state
        assert st is not None

        delta_b = st.S - st.bid
        delta_a = st.ask - st.S

        lambda_b = self.A * np.exp(-self.k * delta_b)
        lambda_a = self.A * np.exp(-self.k * delta_a)

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

        # bid hit: we buy — inventory up, cash down by bid
        if bid_fill:  # buy
            st.q += 1
            st.X -= st.bid
            st.fills_buy += 1

        # ask hit: we sell — inventory down, cash up by ask
        if ask_fill:  # sell
            st.q -= 1
            st.X += st.ask
            st.fills_sell += 1

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

    def step(self) -> MarketState:
        st = self.state
        assert st is not None

        # advance time
        st.t = round(st.t + self.dt, 10)

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
