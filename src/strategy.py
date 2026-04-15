from dataclasses import dataclass

from market import MarketState


@dataclass
class FixedSpreadBaseline:
    """
    fixed hald spread.
        bid = S_t - c
        ask = S_t + c.
    """

    c: float = 0.5

    def apply(self, st: MarketState) -> None:
        st.bid = st.S - self.c
        st.ask = st.S + self.c
