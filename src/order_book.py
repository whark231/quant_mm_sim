from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid
import heapq


class Side(Enum):
    BID = "bid"
    ASK = "ask"


class OrderStatus(Enum):
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"


@dataclass
class Order:
    price: float
    size: float
    side: Side
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filled_size: float = 0.0
    status: OrderStatus = OrderStatus.OPEN
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    def fill(self, quantity: float) -> None:
        self.filled_size += quantity
        if self.remaining_size >= 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        else:
            self.status = OrderStatus.FILLED

    def cancel(self) -> None:
        self.status = OrderStatus.CANCELLED

    def __repr__(self) -> str:
        return (
            f"Order(id={self.id[:8]}, side={self.side.value}, "
            f"price={self.price}, size={self.size}, filled={self.filled_size}, "
            f"ts={self.timestamp.isoformat()})"
        )
    
    def __lt__(self, other):
        if self.price == other.price:
            return self.timestamp < other.timestamp
        else:
            return self.price < other.price
        
    def earlier_order_than(self, other):
        return self.timestamp < other.timestamp
        
class OrderBook:
    def __init__(self):
        self.bids = []
        self.asks = []
        self.orders = {}

    
    def add_order(self, order):
        self.orders[order.id] = order
        if order.side == Side.BID:
            heapq.heappush(self.bids, (-order.price, order))
            if self.best_bid and self.best_ask:
                assert self.best_ask > self.best_bid, f"Crossed book: bid {self.best_bid} >= ask {self.best_ask}"
        else:
            heapq.heappush(self.asks, order)
            if self.best_bid and self.best_ask:
                assert self.best_ask > self.best_bid, f"Crossed book: bid {self.best_bid} >= ask {self.best_ask}"

    
    @property
    def best_bid(self):
        return None if not self.bids else -self.bids[0][0]
    
    @property
    def best_ask(self):
        return None if not self.asks else self.asks[0].price
    
    @property
    def mid_price(self):
        return None if self.best_ask is None or self.best_bid is None else (self.best_ask + self.best_bid) / 2
    
    def match_orders(self):

        while self.best_bid >= self.best_ask:
            bid = self.best_bid[0][1]
            ask = self.best_ask[0]
            fill_quantity = min(bid.size, ask.size)
            resting_order = bid if bid.earlier_order_than(ask) else ask

            # Execute trade at resting order
            resting_order.fill(fill_quantity)

            if not bid.is_active:
                self.remove_order(bid)
            
            if not ask.is_active:
                self.remove_order(bid)
