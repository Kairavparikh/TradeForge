#!/usr/bin/env python3
"""
Trading System Simulator - Complete Demo Script
This script demonstrates the full functionality of the trading system.
"""

import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Import the C++ order book engine
import orderbook_engine

def demo_order_book_functionality():
    """Demonstrate basic order book functionality"""
    print("=" * 80)
    print("1. ORDER BOOK FUNCTIONALITY DEMO")
    print("=" * 80)
    
    book = orderbook_engine.OrderBook()
    BUY = orderbook_engine.Side.BUY
    SELL = orderbook_engine.Side.SELL
    engine_type = "C++"
    
    print(f"Engine: {engine_type}")
    print("\nSubmitting initial orders...")
    
    # Add some buy orders (bids)
    book.submit_limit_order(BUY, 99.95, 1000)
    book.submit_limit_order(BUY, 99.90, 1500)
    book.submit_limit_order(BUY, 99.85, 2000)
    
    # Add some sell orders (asks)
    book.submit_limit_order(SELL, 100.05, 1200)
    book.submit_limit_order(SELL, 100.10, 1800)
    book.submit_limit_order(SELL, 100.15, 2500)
    
    # Show market state
    print(f"\nMarket State:")
    print(f"  Best Bid: ${book.get_best_bid():.2f}")
    print(f"  Best Ask: ${book.get_best_ask():.2f}")
    print(f"  Mid Price: ${book.get_mid_price():.2f}")
    print(f"  Spread: ${book.get_best_ask() - book.get_best_bid():.2f}")
    
    # Show order book depth
    print(f"\nBid Levels:")
    for price, qty in book.get_bid_levels(3):
        print(f"  ${price:.2f} - {qty:,} shares")
    
    print(f"\nAsk Levels:")
    for price, qty in book.get_ask_levels(3):
        print(f"  ${price:.2f} - {qty:,} shares")
    
    # Execute a market order to generate trades
    print(f"\nExecuting market buy order for 2,500 shares...")
    book.submit_market_order(BUY, 2500)
    
    # Show trades
    trades = book.get_trades()
    print(f"\nTrades executed: {len(trades)}")
    for i, trade in enumerate(trades[-3:], 1):  # Show last 3 trades
        print(f"  Trade {i}: {trade.quantity} @ ${trade.price:.2f}")
    
    print(f"\nTotal statistics:")
    print(f"  Orders processed: {book.get_total_orders():,}")
    print(f"  Trades executed: {book.get_total_trades():,}")
    print(f"  Volume traded: ${book.get_total_volume():,.2f}")
    print(f"  Active orders: {book.get_active_orders_count():,}")
    
    return book, engine_type


def demo_performance_benchmark():
    """Demonstrate performance benchmarking using only the C++ engine"""
    print("\n" + "=" * 80)
    print("2. PERFORMANCE BENCHMARK DEMO")
    print("=" * 80)
    
    test_sizes = [1000, 5000, 10000]
    engines = [("C++", lambda: orderbook_engine.OrderBook(), orderbook_engine.Side)]
    
    results = {}
    
    for n_orders in test_sizes:
        print(f"\nBenchmarking with {n_orders:,} orders:")
        print("-" * 40)
        
        # Generate test orders
        np.random.seed(42)
        test_orders = []
        for _ in range(n_orders):
            side = np.random.choice(['BUY', 'SELL'])
            price = np.random.uniform(99.0, 101.0)
            quantity = np.random.randint(10, 1000)
            test_orders.append((side, price, quantity))
        
        for engine_name, engine_factory, side_enum in engines:
            book = engine_factory()
            
            start_time = time.time()
            for side_str, price, quantity in test_orders:
                side = side_enum.BUY if side_str == 'BUY' else side_enum.SELL
                book.submit_limit_order(side, price, quantity)
            end_time = time.time()
            
            execution_time_ms = (end_time - start_time) * 1000
            orders_per_sec = n_orders / (execution_time_ms / 1000)
            
            results[f"{engine_name}_{n_orders}"] = {
                'time_ms': execution_time_ms,
                'orders_per_sec': orders_per_sec,
                'trades': book.get_total_trades()
            }
            
            print(f"  {engine_name:>6}: {execution_time_ms:>8.2f}ms | "
                  f"{orders_per_sec:>10,.0f} orders/sec | "
                  f"{book.get_total_trades():>6,} trades")
    
    return results


def demo_trading_strategies():
    """Demonstrate trading strategies"""
    print("\n" + "=" * 80)
    print("3. TRADING STRATEGIES DEMO")
    print("=" * 80)
    
    from trading_strategies import VWAPStrategy, RandomizedStrategy, MovingAverageStrategy
    
    book = orderbook_engine.OrderBook()
    BUY = orderbook_engine.Side.BUY
    SELL = orderbook_engine.Side.SELL
    
    # Add liquidity
    print("\nInitializing market with liquidity...")
    base_price = 100.0
    for i in range(5):
        book.submit_limit_order(BUY, base_price - (i+1)*0.01, 500)
        book.submit_limit_order(SELL, base_price + (i+1)*0.01, 500)
    
    print(f"Market initialized - Mid price: ${book.get_mid_price():.2f}")
    
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.randn(50) * 0.1)
    market_data = pd.DataFrame({
        'price': prices,
        'volume': np.random.randint(100, 1000, 50)
    })
    
    strategies = [
        VWAPStrategy(target_quantity=5000, target_side="BUY"),
        RandomizedStrategy(target_quantity=5000, target_side="BUY"),
        MovingAverageStrategy(position_size=4000)
    ]
    
    results = []
    for strategy in strategies:
        print(f"\nTesting {strategy.name} Strategy:")
        print("-" * 30)
        book.clear()
        for i in range(5):
            book.submit_limit_order(BUY, base_price - (i+1)*0.01, 500)
            book.submit_limit_order(SELL, base_price + (i+1)*0.01, 500)
        result = strategy.execute(book, market_data)
        results.append(result)
        print(f"  Orders submitted: {result.orders_submitted}")
        print(f"  Trades executed: {result.trades_count}")
        print(f"  Fill rate: {result.fill_rate:.1%}")
        print(f"  Volume traded: {result.volume_traded:,.0f}")
        print(f"  Avg execution price: ${result.avg_execution_price:.3f}")
        print(f"  Execution cost: ${result.execution_cost:.2f}")
        print(f"  Slippage: ${result.slippage:.4f}")
    
    return results


def demo_analysis_and_visualization():
    """Demonstrate analysis capabilities"""
    print("\n" + "=" * 80)
    print("4. ANALYSIS & VISUALIZATION DEMO")
    print("=" * 80)
    
    from backtesting_engine import BacktestingEngine
    import matplotlib.pyplot as plt
    
    engine = BacktestingEngine()
    print("Executing strategies with 5,000 orders...")
    results_df = engine.run_strategy_comparison(n_orders=5000)
    report = engine.generate_analysis_report()
    print(report)
    
    # Visualization
    if not results_df.empty and 'fill_rate' in results_df.columns:
        plt.figure(figsize=(10, 6))
        fill_rates = results_df.groupby('strategy_name')['fill_rate'].mean()
        plt.subplot(1, 2, 1)
        bars = plt.bar(fill_rates.index, fill_rates.values)
        plt.title('Average Fill Rate by Strategy')
        plt.ylabel('Fill Rate')
        plt.xticks(rotation=45)
        for bar, value in zip(bars, fill_rates.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{value:.1%}', ha='center', va='bottom')
        exec_costs = results_df.groupby('strategy_name')['execution_cost'].mean()
        plt.subplot(1, 2, 2)
        bars = plt.bar(exec_costs.index, exec_costs.values)
        plt.title('Average Execution Cost by Strategy')
        plt.ylabel('Execution Cost ($)')
        plt.xticks(rotation=45)
        for bar, value in zip(bars, exec_costs.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(exec_costs.values)*0.01,
                     f'${value:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
        print("\n📊 Basic visualization saved as 'demo_results.png'")
        plt.show()
    
    return results_df, engine.performance_data


def main():
    print("🚀 TRADING SYSTEM SIMULATOR - COMPLETE DEMO")
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        book, engine_type = demo_order_book_functionality()
        perf_results = demo_performance_benchmark()
        strategy_results = demo_trading_strategies()
        analysis_results, analysis_perf = demo_analysis_and_visualization()
        print("\nDemo completed successfully!")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
