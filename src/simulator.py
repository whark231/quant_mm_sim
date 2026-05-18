from src.order_book import OrderBook, Order, Side
from src.market_maker import MarketMaker
import pandas as pd


class Simulator():
    def __init__(self, mm: MarketMaker, book: OrderBook, df: pd.DataFrame):
        self.mm = mm
        self.book = book
        self.df = df
        self.record = {
            'Time': [],
            'Mid_Price': [],
            'Reservation_Price' : [],
            'Bid' : [],
            'Ask' : [],
            'Position' : [],
            'PnL' : [],
            'Spread' : []
        }

    # Records performance during simulation
    def record_state(self, i):
        bid, ask = self.mm.get_quotes()
        self.record['Time'].append(i)
        self.record['Mid_Price'].append(self.book.mid_price)
        self.record['Reservation_Price'].append(self.mm.reservation_price)
        self.record['Bid'].append(bid)
        self.record['Ask'].append(ask)
        self.record['Position'].append(self.mm.position)
        self.record['PnL'].append(self.mm.pnl)
        self.record['Spread'].append(ask - bid)

    # Returns results of trade strategy
    def get_results(self):
        return pd.DataFrame(self.record).set_index('Time')
    
    # Add historical trade to the order book -> moves mid-price
    def update_market(self, i):
        row = self.df.iloc[i]
        side = Side.BID if row['side'] == 'buy' else Side.ASK
        order = Order(row['price'], row['size'], side)

        self.book.add_order(order)

    # Cancels Market Maker's stale quotes
    def cancel_old_quotes(self):
        if self.mm.current_bid_id is not None:
            self.book.orders[self.mm.current_bid_id].cancel()
        if self.mm.current_ask_id is not None:
            self.book.orders[self.mm.current_ask_id].cancel()
        self.mm.cancel_quotes()

    # Posts new quotes
    def post_new_quotes(self):
        bid, ask = self.mm.get_quotes()
        bid_order = Order(price=bid, size=0.01, side=Side.BID)
        ask_order = Order(price=ask, size=0.01, side=Side.ASK)

        self.book.add_order(bid_order)
        self.book.add_order(ask_order)

        self.mm.current_ask_id = ask_order.id
        self.mm.current_bid_id = bid_order.id

    # Updates Market Maker's inventory and cash if market moves through Market Maker's quotes
    def check_for_fills(self, i):
        trade = self.df.iloc[i]

        if self.mm.current_bid_id is not None:
            bid_order = self.book.orders[self.mm.current_bid_id]

            if trade['side'] == 'sell' and trade['price'] <= bid_order.price:
                self.mm.update_inventory(Side.BID, min(trade['size'], bid_order.remaining_size), trade['price'])
        
        if self.mm.current_ask_id is not None:
            ask_order = self.book.orders[self.mm.current_ask_id]

            if trade['side'] == 'buy' and trade['price'] >= ask_order.price:
                self.mm.update_inventory(Side.ASK, min(trade['size'], ask_order.remaining_size), trade['price'])

    # populate order book with a few orders
    def warmup(self, n=20):
        for i in range(n):
            row = self.df.iloc[i]
            side = Side.BID if row['side'] == 'buy' else Side.ASK
            order = Order(price=row['price'], size=row['size'], side=side)
            self.book.add_order(order)
        
    # runs the simulation
    def run(self, warmup_periods=20):
        self.warmup(warmup_periods)
        for i in range(len(self.df)):
            self.update_market(i)
            self.cancel_old_quotes()
            self.post_new_quotes()
            self.check_for_fills(i)
            self.record_state(i)



