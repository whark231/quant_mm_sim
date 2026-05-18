import math
from src.order_book import OrderBook, Side
from datetime import datetime, timezone


class MarketMaker():
    def __init__(self, starting_wealth, order_book: OrderBook, gamma = 0.1, sigma = 2.0, kappa = 1.5):
        self.cash = starting_wealth
        self.order_book = order_book
        self.position = 0.0
        self.gamma = gamma # risk aversion
        self.sigma = sigma # per tick vol
        self.kappa = kappa # order arrival sensitvity
        self.session_start = None
        self.session_duration_hours = 8.0 # trading day length
        self.current_bid_id = None
        self.current_ask_id = None


    @property 
    def pnl(self) -> float:
        return self.cash + self.position * self.order_book.mid_price

    @property
    def dynamic_gamma(self) -> float:
        return self.gamma
    
    @property
    def time_remaining(self) -> float:
        if self.session_start is None:
            return 1.0
        
        elapsed = (datetime.now(timezone.utc) - self.session_start).total_seconds()
        elapsed_fraction = elapsed / (self.session_duration_hours * 3600)
        return max(0.0, 1.0 - elapsed_fraction)

    def reservation_price(self, mid_price: float) -> float:
        # How much we discount the mid_price based on inventory risk
        inventory_penalty = self.position * self.gamma * self.sigma**2 * self.time_remaining
        return mid_price - inventory_penalty
    
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

    def get_quotes(self, mid_price: float):
        # Compensation for holding risk over remaining time
        risk_term = self.gamma * self.sigma**2 * self.time_remaining

        # Compensation for providing liquidity given order arrival rate
        liquidity_term = (2 / self.gamma) * math.log(1 + self.gamma / self.kappa)

        spread =  risk_term + liquidity_term

        bid = self.reservation_price(mid_price) - spread / 2
        ask = self.reservation_price(mid_price) + spread / 2
        return bid, ask
    
    def print_quotes(self) -> None:
        bid, ask = self.get_quotes()
        print(
            f"Reservation Price: ${self.reservation_price:,.2f}\n"
            f"Bid              : ${bid:,.2f}\n"
            f"Ask              : ${ask:,.2f}"
        )

    def start_session(self):
        self.session_start = datetime.now(timezone.utc)