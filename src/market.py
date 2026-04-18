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
    fills_buy: int = 0   # bid hit: buy
    fills_sell: int = 0  # ask hit: sell

    # order-flow / LOB proxies for alpha signal
    ofi_window: list = field(default_factory=list)  # recent signed aggressive flow
    bid_depth: float = 80.0
    ask_depth: float = 80.0
    last_alpha: float = 0.0

    #history
    hist_t: list = field(default_factory=list)
    hist_S: list = field(default_factory=list)
    hist_bid: list = field(default_factory=list)
    hist_ask: list = field(default_factory=list)
    hist_q: list = field(default_factory=list) # inventory
    hist_X: list = field(default_factory=list) # cash
    hist_W: list = field(default_factory=list) # market wealth
    hist_alpha: list = field(default_factory=list)

    @property
    def wealth(self) -> float:
        """W_t = X_t + q_t * S_t"""
        return self.X + self.q * self.S

    @property
    def cumulative_fills(self) -> int:
        return self.fills_buy + self.fills_sell




