from dataclasses import dataclass, field
import numpy as np

@dataclass
class MarketState:
    #time
    t: float = 0.0
    #mid price
    S: float = 100.0
    #inventory
    q: int = 0
    #cash
    X: float = 0.0
    bid: float = None
    ask: float = None
    fills_buy: int = 0   # bid hit: uy
    fills_sell: int = 0  # ask hit: sell

    #history
    hist_t: list = field(default_factory=list)
    hist_S: list = field(default_factory=list)
    hist_bid: list = field(default_factory=list)
    hist_ask: list = field(default_factory=list)
    hist_q: list = field(default_factory=list) # inventory
    hist_X: list = field(default_factory=list) # cash
    hist_W: list = field(default_factory=list) # market wealth

    @property
    def wealth(self) -> float:
        """W_t = X_t + q_t * S_t"""
        return self.X + self.q * self.S

    @property
    def cumulative_fills(self) -> int:
        return self.fills_buy + self.fills_sell




