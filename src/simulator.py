from src.order_book import OrderBook, Order, Side
from src.market_maker import MarketMaker
import pandas as pd


class Simulator():
    def __init__(self, mm: MarketMaker, book: OrderBook, df: pd.DataFrame):
        self.mm = mm
        self.book = book
        self.df = df
        self.iteration = 0
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

    def record_state(self):
        bid, ask = self.mm.get_quotes()
        self.record['Time'].append(self.iteration)
        self.record['Mid_Price'].append(self.book.mid_price)
        self.record['Reservation_Price'].append(self.mm.reservation_price)
        self.record['Bid'].append(bid)
        self.record['Ask'].append(ask)
        self.record['Position'].append(self.mm.position)
        self.record['PnL'].append(self.mm.pnl)
        self.record['Spread'].append(ask - bid)

    def get_results(self):
        return pd.DataFrame(self.record).set_index('Time')
    
    def update_market(self):
        row = self.df.iloc[self.iteration]
        side = Side.BID if row['side'] == 'buy' else Side.ASK
        order = Order(row['price'], row['size'], side)

        self.book.add_order(order)

        self.iteration += 1

    
    def cancel_old_quotes(self):
        if self.mm.current_bid_id is not None:
            self.book.orders[self.mm.current_bid_id].cancel()
        if self.mm.current_ask_id is not None:
            self.book.orders[self.mm.current_ask_id].cancel()
        self.mm.cancel_quotes()

    def post_new_quotes(self):
        
        bid, ask = self.mm.get_quotes()
        bid_order = Order(price=bid, size=0.01, side=Side.BID)
        ask_order = Order(price=ask, size=0.01, side=Side.ASK)

        self.book.add_order(bid_order)
        self.book.add_order(ask_order)

        self.mm.current_ask_id = ask_order.id
        self.mm.current_bid_id = bid_order.id





