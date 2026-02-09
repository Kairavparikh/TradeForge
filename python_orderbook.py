"""
Pure Python implementation of Order Book for performance comparison
"""
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


@dataclass
class Order:
    id: int
    side: Side
    order_type: OrderType
    price: float
    quantity: int
    filled_quantity: int = 0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class Trade:
    buy_order_id: int
    sell_order_id: int
    price: float
    quantity: int
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class PythonOrderBook:
    """Pure Python implementation of limit order book"""
    
    def __init__(self):
        # Buy side: higher prices first
        self.buy_levels = {}  # price -> deque of orders
        self.buy_prices = []  # sorted list of prices (descending)
        
        # Sell side: lower prices first  
        self.sell_levels = {}  # price -> deque of orders
        self.sell_prices = []  # sorted list of prices (ascending)
        
        self.active_orders = {}  # order_id -> order
        self.trades = []
        self.next_order_id = 1
        
        # Statistics
        self.total_orders_processed = 0
        self.total_trades_executed = 0
        self.total_volume_traded = 0.0
    
    def submit_limit_order(self, side: Side, price: float, quantity: int) -> int:
        """Submit a limit order"""
        order = Order(
            id=self.next_order_id,
            side=side,
            order_type=OrderType.LIMIT,
            price=price,
            quantity=quantity
        )
        self.next_order_id += 1
        self.active_orders[order.id] = order
        self.total_orders_processed += 1
        
        self._match_order(order)
        
        # If not fully filled, add to book
        if order.filled_quantity < order.quantity:
            self._add_to_book(order)
        else:
            # Remove fully filled order
            del self.active_orders[order.id]
        
        return order.id
    
    def submit_market_order(self, side: Side, quantity: int) -> int:
        """Submit a market order"""
        order = Order(
            id=self.next_order_id,
            side=side,
            order_type=OrderType.MARKET,
            price=0.0,
            quantity=quantity
        )
        self.next_order_id += 1
        self.total_orders_processed += 1
        
        self._match_order(order)
        return order.id
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel an order"""
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        del self.active_orders[order_id]
        
        # Remove from book
        if order.side == Side.BUY:
            levels = self.buy_levels
            prices = self.buy_prices
        else:
            levels = self.sell_levels
            prices = self.sell_prices
        
        if order.price in levels:
            level = levels[order.price]
            # Remove order from level (inefficient but simple for demo)
            new_level = deque()
            for o in level:
                if o.id != order_id:
                    new_level.append(o)
            
            if new_level:
                levels[order.price] = new_level
            else:
                del levels[order.price]
                prices.remove(order.price)
        
        return True
    
    def _add_to_book(self, order: Order):
        """Add order to the appropriate side of the book"""
        if order.side == Side.BUY:
            if order.price not in self.buy_levels:
                self.buy_levels[order.price] = deque()
                self.buy_prices.append(order.price)
                self.buy_prices.sort(reverse=True)  # Descending for bids
            self.buy_levels[order.price].append(order)
        else:
            if order.price not in self.sell_levels:
                self.sell_levels[order.price] = deque()
                self.sell_prices.append(order.price)
                self.sell_prices.sort()  # Ascending for asks
            self.sell_levels[order.price].append(order)
    
    def _match_order(self, order: Order):
        """Try to match incoming order against opposite side"""
        if order.side == Side.BUY:
            opposite_levels = self.sell_levels
            opposite_prices = self.sell_prices
        else:
            opposite_levels = self.buy_levels
            opposite_prices = self.buy_prices
        
        while order.filled_quantity < order.quantity and opposite_prices:
            best_price = opposite_prices[0]
            
            # Check if we can match
            can_match = False
            if order.order_type == OrderType.MARKET:
                can_match = True
            elif order.side == Side.BUY and order.price >= best_price:
                can_match = True
            elif order.side == Side.SELL and order.price <= best_price:
                can_match = True
            
            if not can_match:
                break
            
            # Match against orders at this price level
            level = opposite_levels[best_price]
            while level and order.filled_quantity < order.quantity:
                matching_order = level[0]
                
                # Skip cancelled orders
                if matching_order.id not in self.active_orders:
                    level.popleft()
                    continue
                
                trade_quantity = min(
                    order.quantity - order.filled_quantity,
                    matching_order.quantity - matching_order.filled_quantity
                )
                
                # Execute trade
                if order.side == Side.BUY:
                    buy_order, sell_order = order, matching_order
                else:
                    buy_order, sell_order = matching_order, order
                
                trade = Trade(
                    buy_order_id=buy_order.id,
                    sell_order_id=sell_order.id,
                    price=best_price,
                    quantity=trade_quantity
                )
                
                self.trades.append(trade)
                self.total_trades_executed += 1
                self.total_volume_traded += best_price * trade_quantity
                
                # Update filled quantities
                order.filled_quantity += trade_quantity
                matching_order.filled_quantity += trade_quantity
                
                # Remove fully filled order
                if matching_order.filled_quantity == matching_order.quantity:
                    del self.active_orders[matching_order.id]
                    level.popleft()
            
            # Remove empty price level
            if not level:
                del opposite_levels[best_price]
                opposite_prices.pop(0)
    
    def get_best_bid(self) -> float:
        """Get best bid price"""
        return self.buy_prices[0] if self.buy_prices else 0.0
    
    def get_best_ask(self) -> float:
        """Get best ask price"""
        return self.sell_prices[0] if self.sell_prices else 0.0
    
    def get_mid_price(self) -> float:
        """Get mid price"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return 0.0
    
    def get_bid_levels(self, depth: int = 5) -> List[Tuple[float, int]]:
        """Get bid levels with depth"""
        levels = []
        for i, price in enumerate(self.buy_prices[:depth]):
            total_qty = sum(o.quantity - o.filled_quantity 
                          for o in self.buy_levels[price])
            levels.append((price, total_qty))
        return levels
    
    def get_ask_levels(self, depth: int = 5) -> List[Tuple[float, int]]:
        """Get ask levels with depth"""
        levels = []
        for i, price in enumerate(self.sell_prices[:depth]):
            total_qty = sum(o.quantity - o.filled_quantity 
                          for o in self.sell_levels[price])
            levels.append((price, total_qty))
        return levels
    
    def get_trades(self) -> List[Trade]:
        """Get all trades"""
        return self.trades
    
    def get_recent_trades(self, count: int) -> List[Trade]:
        """Get recent trades"""
        return self.trades[-count:] if len(self.trades) >= count else self.trades
    
    def get_total_orders(self) -> int:
        return self.total_orders_processed
    
    def get_total_trades(self) -> int:
        return self.total_trades_executed
    
    def get_total_volume(self) -> float:
        return self.total_volume_traded
    
    def get_active_orders_count(self) -> int:
        return len(self.active_orders)
    
    def clear(self):
        """Clear all data"""
        self.buy_levels.clear()
        self.buy_prices.clear()
        self.sell_levels.clear()
        self.sell_prices.clear()
        self.active_orders.clear()
        self.trades.clear()
        self.next_order_id = 1
        self.total_orders_processed = 0
        self.total_trades_executed = 0
        self.total_volume_traded = 0.0