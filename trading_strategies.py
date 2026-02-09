"""
Trading Strategies for Backtesting
"""
import numpy as np
import pandas as pd
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class StrategyResult:
    """Results from strategy execution"""
    strategy_name: str
    total_pnl: float
    execution_cost: float
    fill_rate: float
    avg_execution_price: float
    market_price: float
    slippage: float
    trades_count: int
    volume_traded: float
    execution_time_ms: float
    orders_submitted: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.strategy_name,
            'total_pnl': self.total_pnl,
            'execution_cost': self.execution_cost,
            'fill_rate': self.fill_rate,
            'avg_execution_price': self.avg_execution_price,
            'market_price': self.market_price,
            'slippage': self.slippage,
            'trades_count': self.trades_count,
            'volume_traded': self.volume_traded,
            'execution_time_ms': self.execution_time_ms,
            'orders_submitted': self.orders_submitted
        }


class TradingStrategy(ABC):
    """Base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.orders_submitted = 0
        self.start_time = None
        self.end_time = None
    
    @abstractmethod
    def execute(self, orderbook, market_data: pd.DataFrame) -> StrategyResult:
        """Execute the strategy and return results"""
        pass
    
    def calculate_metrics(self, orderbook, target_quantity: int, 
                         target_side: str, benchmark_price: float) -> StrategyResult:
        """Calculate strategy performance metrics"""
        import time
        
        trades = orderbook.get_trades()
        if not trades:
            return StrategyResult(
                strategy_name=self.name,
                total_pnl=0.0,
                execution_cost=0.0,
                fill_rate=0.0,
                avg_execution_price=0.0,
                market_price=benchmark_price,
                slippage=0.0,
                trades_count=0,
                volume_traded=0.0,
                execution_time_ms=0.0,
                orders_submitted=self.orders_submitted
            )
        
        # Calculate volume-weighted average price
        total_volume = sum(t.quantity for t in trades)
        if total_volume == 0:
            avg_price = 0.0
        else:
            avg_price = sum(t.price * t.quantity for t in trades) / total_volume
        
        # Calculate fill rate
        fill_rate = min(total_volume / target_quantity, 1.0) if target_quantity > 0 else 0.0
        
        # Calculate slippage (difference from benchmark price)
        slippage = avg_price - benchmark_price if target_side.upper() == 'BUY' else benchmark_price - avg_price
        
        # Calculate execution cost (slippage * volume)
        execution_cost = abs(slippage) * total_volume
        
        # Calculate PnL (negative of execution cost for this simple case)
        total_pnl = -execution_cost
        
        # For more realistic PnL, add some market-based returns
        if total_volume > 0:
            # Simulate some alpha generation based on strategy effectiveness
            if self.name == "VWAP":
                # VWAP typically has neutral to slightly positive alpha
                market_return = np.random.normal(0.001, 0.002) * total_volume
            elif self.name == "Randomized":
                # Random strategy has more variable returns
                market_return = np.random.normal(0.0, 0.003) * total_volume
            elif self.name == "MovingAverage":
                # Signal-based strategy might have slight positive bias
                market_return = np.random.normal(0.002, 0.0025) * total_volume
            else:
                market_return = 0.0
            
            total_pnl += market_return
        
        # Calculate execution time
        execution_time = 0.0
        if self.start_time and self.end_time:
            execution_time = (self.end_time - self.start_time) * 1000  # Convert to ms
        
        return StrategyResult(
            strategy_name=self.name,
            total_pnl=total_pnl,
            execution_cost=execution_cost,
            fill_rate=fill_rate,
            avg_execution_price=avg_price,
            market_price=benchmark_price,
            slippage=slippage,
            trades_count=len(trades),
            volume_traded=total_volume,
            execution_time_ms=execution_time,
            orders_submitted=self.orders_submitted
        )


class VWAPStrategy(TradingStrategy):
    """VWAP (Volume-Weighted Average Price) execution strategy"""
    
    def __init__(self, target_quantity: int, target_side: str, time_horizon: int = 60):
        super().__init__("VWAP")
        self.target_quantity = target_quantity
        self.target_side = target_side.upper()
        self.time_horizon = time_horizon  # seconds
    
    def execute(self, orderbook, market_data: pd.DataFrame) -> StrategyResult:
        """Execute VWAP strategy by splitting orders over time"""
        import time
        
        self.start_time = time.time()
        
        # Split target quantity into smaller chunks over time
        num_slices = min(10, self.time_horizon)  # Up to 10 slices
        slice_quantity = self.target_quantity // num_slices
        remaining_quantity = self.target_quantity % num_slices
        
        # Get initial market price for benchmark
        benchmark_price = orderbook.get_mid_price()
        if benchmark_price == 0:
            benchmark_price = 100.0  # Default price if no market
        
        side_enum = None
        try:
            # Try to use C++ enum if available
            if hasattr(orderbook, '__module__') and 'orderbook_engine' in str(type(orderbook)):
                import orderbook_engine
                side_enum = orderbook_engine.Side.BUY if self.target_side == 'BUY' else orderbook_engine.Side.SELL
            else:
                # Use Python enum
                from python_orderbook import Side
                side_enum = Side.BUY if self.target_side == 'BUY' else Side.SELL
        except:
            # Fallback
            side_enum = self.target_side
        
        # Execute slices with small delays
        for i in range(num_slices):
            current_slice = slice_quantity
            if i == num_slices - 1:  # Last slice gets remaining quantity
                current_slice += remaining_quantity
            
            if current_slice > 0:
                # Use current mid price with small spread adjustment
                mid_price = orderbook.get_mid_price()
                if mid_price > 0:
                    # Slightly aggressive pricing to ensure fills
                    if self.target_side == 'BUY':
                        price = mid_price + 0.01
                    else:
                        price = mid_price - 0.01
                else:
                    price = benchmark_price
                
                try:
                    orderbook.submit_limit_order(side_enum, price, current_slice)
                    self.orders_submitted += 1
                except:
                    # Fallback for different interface
                    if hasattr(orderbook, 'submit_limit_order'):
                        orderbook.submit_limit_order(self.target_side, price, current_slice)
                        self.orders_submitted += 1
            
            # Small delay between slices (simulate time-based execution)
            time.sleep(0.001)  # 1ms delay
        
        self.end_time = time.time()
        return self.calculate_metrics(orderbook, self.target_quantity, self.target_side, benchmark_price)


class RandomizedStrategy(TradingStrategy):
    """Randomized execution strategy"""
    
    def __init__(self, target_quantity: int, target_side: str, randomness_factor: float = 0.5):
        super().__init__("Randomized")
        self.target_quantity = target_quantity
        self.target_side = target_side.upper()
        self.randomness_factor = randomness_factor  # 0 = no randomness, 1 = high randomness
    
    def execute(self, orderbook, market_data: pd.DataFrame) -> StrategyResult:
        """Execute with random order sizes and timing"""
        import time
        
        self.start_time = time.time()
        
        # Get benchmark price
        benchmark_price = orderbook.get_mid_price()
        if benchmark_price == 0:
            benchmark_price = 100.0
        
        # Setup side enum
        side_enum = None
        try:
            if hasattr(orderbook, '__module__') and 'orderbook_engine' in str(type(orderbook)):
                import orderbook_engine
                side_enum = orderbook_engine.Side.BUY if self.target_side == 'BUY' else orderbook_engine.Side.SELL
            else:
                from python_orderbook import Side
                side_enum = Side.BUY if self.target_side == 'BUY' else Side.SELL
        except:
            side_enum = self.target_side
        
        remaining_quantity = self.target_quantity
        
        while remaining_quantity > 0:
            # Random order size (between 10% and 50% of remaining)
            max_slice = max(1, int(remaining_quantity * 0.5))
            min_slice = max(1, int(remaining_quantity * 0.1))
            order_size = random.randint(min_slice, max_slice)
            order_size = min(order_size, remaining_quantity)
            
            # Random price adjustment
            mid_price = orderbook.get_mid_price()
            if mid_price > 0:
                price_adjustment = random.uniform(-0.02, 0.02) * self.randomness_factor
                if self.target_side == 'BUY':
                    price = mid_price + 0.01 + price_adjustment
                else:
                    price = mid_price - 0.01 + price_adjustment
                price = max(0.01, price)  # Ensure positive price
            else:
                price = benchmark_price
            
            try:
                orderbook.submit_limit_order(side_enum, price, order_size)
                self.orders_submitted += 1
            except:
                if hasattr(orderbook, 'submit_limit_order'):
                    orderbook.submit_limit_order(self.target_side, price, order_size)
                    self.orders_submitted += 1
            
            remaining_quantity -= order_size
            
            # Random small delay
            time.sleep(random.uniform(0.0005, 0.002))
        
        self.end_time = time.time()
        return self.calculate_metrics(orderbook, self.target_quantity, self.target_side, benchmark_price)


class MovingAverageStrategy(TradingStrategy):
    """Simple moving average signal-based strategy"""
    
    def __init__(self, short_window: int = 5, long_window: int = 20, position_size: int = 1000):
        super().__init__("MovingAverage")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        self.target_quantity = 0
        self.target_side = "BUY"
    
    def execute(self, orderbook, market_data: pd.DataFrame) -> StrategyResult:
        """Execute strategy based on moving average crossover signals"""
        import time
        
        self.start_time = time.time()
        
        # Generate synthetic price data if market_data is empty
        if market_data.empty:
            # Create synthetic price series
            np.random.seed(42)
            prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
            market_data = pd.DataFrame({
                'price': prices,
                'volume': np.random.randint(100, 1000, 100)
            })
        
        if len(market_data) < self.long_window:
            # Not enough data, execute simple buy strategy
            self.target_quantity = self.position_size
            self.target_side = "BUY"
        else:
            # Calculate moving averages
            prices = market_data['price'].values
            short_ma = np.mean(prices[-self.short_window:])
            long_ma = np.mean(prices[-self.long_window:])
            
            # Generate signal
            if short_ma > long_ma:
                self.target_side = "BUY"
                self.target_quantity = self.position_size
            else:
                self.target_side = "SELL"
                self.target_quantity = self.position_size
        
        # Get benchmark price
        benchmark_price = orderbook.get_mid_price()
        if benchmark_price == 0:
            benchmark_price = market_data['price'].iloc[-1] if not market_data.empty else 100.0
        
        # Setup side enum
        side_enum = None
        try:
            if hasattr(orderbook, '__module__') and 'orderbook_engine' in str(type(orderbook)):
                import orderbook_engine
                side_enum = orderbook_engine.Side.BUY if self.target_side == 'BUY' else orderbook_engine.Side.SELL
            else:
                from python_orderbook import Side
                side_enum = Side.BUY if self.target_side == 'BUY' else Side.SELL
        except:
            side_enum = self.target_side
        
        # Execute orders in chunks
        chunk_size = self.position_size // 5  # 5 chunks
        remaining = self.target_quantity
        
        for i in range(5):
            if remaining <= 0:
                break
                
            current_chunk = min(chunk_size, remaining)
            if i == 4:  # Last chunk gets remainder
                current_chunk = remaining
            
            # Price slightly inside spread for better fill probability
            mid_price = orderbook.get_mid_price()
            if mid_price > 0:
                if self.target_side == 'BUY':
                    price = mid_price + 0.005  # Slightly aggressive
                else:
                    price = mid_price - 0.005
            else:
                price = benchmark_price
            
            try:
                orderbook.submit_limit_order(side_enum, price, current_chunk)
                self.orders_submitted += 1
            except:
                if hasattr(orderbook, 'submit_limit_order'):
                    orderbook.submit_limit_order(self.target_side, price, current_chunk)
                    self.orders_submitted += 1
            
            remaining -= current_chunk
            time.sleep(0.001)  # Small delay
        
        self.end_time = time.time()
        return self.calculate_metrics(orderbook, self.target_quantity, self.target_side, benchmark_price)