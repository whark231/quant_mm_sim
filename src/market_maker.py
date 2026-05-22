import math
from src.order_book import OrderBook, Side
from datetime import datetime, timezone


class MarketMaker():
    def __init__(self, starting_wealth, order_book: OrderBook, gamma=0.1, sigma=2.0, kappa=1.5, alpha=1.0, beta=1.0):
        self.cash = starting_wealth
        self.starting_weatlth = starting_wealth
        self.order_book = order_book
        self.position = 0.0
        self.gamma = gamma # risk aversion
        self.sigma = sigma # per tick vol
        self.kappa = kappa # order arrival sensitvity
        self.alpha = alpha
        self.beta = beta
        self.session_start = None
        self.session_duration_hours = 8.0 # trading day length
        self.current_bid_id = None
        self.current_ask_id = None
        self.is_quoting = True


    @property
    def dynamic_gamma(self) -> float:
        # alpha, beta ~ risk-aversion as inventory builds and time runs out respectively
        return self.gamma * (1+ self.alpha * self.position**2) * (1 + self.beta * (1 - self.time_remaining))
    
    @property
    def time_remaining(self) -> float:
        if self.session_start is None:
            return 1.0
        
        elapsed = (datetime.now(timezone.utc) - self.session_start).total_seconds()
        elapsed_fraction = elapsed / (self.session_duration_hours * 3600)
        return max(0.0, 1.0 - elapsed_fraction)

    def reservation_price(self, mid_price: float, dynamic_sigma) -> float:
        # How much we discount the mid_price based on inventory risk
        inventory_penalty = self.position * self.dynamic_gamma * dynamic_sigma**2 * self.time_remaining
        return mid_price - inventory_penalty
    
    def pnl(self, mid_price: float) -> float:
        return (self.cash + self.position * mid_price) - self.starting_weatlth
    
    def update_inventory(self, side: Side, filled_size: float, filled_price: float):
        if side == Side.BID:
            self.position += filled_size
            self.cash -= filled_size * filled_price
        else:
            self.position -= filled_size
            self.cash += filled_size * filled_price

    def cancel_quotes(self):
        self.current_bid_ask = None
        self.current_bid_ask = None

    def get_quotes(self, mid_price: float, dynamic_sigma: float = 2.0, ofi: float = 0.8, scaling_factor: float = 2.0):
        # Compensation for holding risk over remaining time
        risk_term = self.dynamic_gamma * dynamic_sigma**2 * self.time_remaining

        # Compensation for providing liquidity given order arrival rate
        liquidity_term = (2 / self.dynamic_gamma) * math.log(1 + self.dynamic_gamma / self.kappa)

        base_spread =  risk_term + liquidity_term

        ofi_multiplier = 1 + scaling_factor * abs(ofi)
        spread = base_spread * ofi_multiplier

        r = self.reservation_price(mid_price, dynamic_sigma)
        bid = r - spread / 2
        ask = r + spread / 2
        return bid, ask
    
    def print_quotes(self, mid_price=None):
        if mid_price is None:
            mid_price = self.order_book.mid_price
        if mid_price is None:
            print("No mid-price available — order book is empty")
            return
        bid, ask = self.get_quotes(mid_price, self.dynamic_sigma if hasattr(self, 'dynamic_sigma') else 2.0)
        print(f"Bid: {bid:.2f}")
        print(f"Ask: {ask:.2f}")
        print(f"Reservation Price: {self.reservation_price(mid_price, self.dynamic_sigma if hasattr(self, 'dynamic_sigma') else 2.0):.2f}")

    def start_session(self):
        self.session_start = datetime.now(timezone.utc)