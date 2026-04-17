from dataclasses import dataclass

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
