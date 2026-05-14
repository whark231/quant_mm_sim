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
    filled_price: float = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filled_size: float = 0.0
    status: OrderStatus = OrderStatus.OPEN
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    arrival_time: datetime = None

    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    def fill(self, quantity: float) -> None:
        self.filled_size += quantity
        if self.remaining_size > 0:
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
        
    def earlier_arrival_than(self, other):
        return self.arrival_time < other.arrival_time
        
class OrderBook:
    def __init__(self):
        self.bids = []
        self.asks = []
        self.orders = {}

    
    def add_order(self, order):
        self.orders[order.id] = order
        order.arrival_time = datetime.now(timezone.utc)
        if order.side == Side.BID:
            heapq.heappush(self.bids, (-order.price, order))
        else:
            heapq.heappush(self.asks, order)
    
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

        while self.best_bid is not None and self.best_ask is not None and self.best_bid >= self.best_ask:

            while self.bids and not self.bids[0][1].is_active:
                heapq.heappop(self.bids)
            while self.asks and not self.asks[0].is_active:
                heapq.heappop(self.asks)
                
            # Re-check after cleanup
            if not self.bids or not self.asks:
                break

            bid = self.bids[0][1]
            ask = self.asks[0]
            fill_quantity = min(bid.size, ask.size)
            resting_order_price = bid.price if bid.earlier_arrival_than(ask) else ask.price

            bid.filled_price = resting_order_price
            ask.filled_price = resting_order_price

            # Execute trades
            bid.fill(fill_quantity)
            ask.fill(fill_quantity)
            
            if self.best_bid is not None and self.best_ask is not None:
                assert self.best_ask > self.best_bid, f"Crossed book after matching: bid {self.best_bid} >= ask {self.best_ask}"
