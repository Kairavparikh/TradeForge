"""
Backtesting Engine with Performance Analysis and Visualization
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random
from datetime import datetime, timedelta

# Import strategies
from trading_strategies import VWAPStrategy, RandomizedStrategy, MovingAverageStrategy, StrategyResult

# Try to import C++ engine, fallback to Python
try:
    import orderbook_engine
    CPP_ENGINE_AVAILABLE = True
except ImportError:
    CPP_ENGINE_AVAILABLE = False

from python_orderbook import PythonOrderBook


class MarketDataGenerator:
    """Generate synthetic market data for backtesting"""
    
    @staticmethod
    def generate_price_series(n_points: int = 1000, initial_price: float = 100.0, 
                            volatility: float = 0.02) -> pd.DataFrame:
        """Generate realistic price series with random walk"""
        np.random.seed(42)  # For reproducible results
        
        # Generate price movements
        returns = np.random.normal(0, volatility, n_points)
        prices = [initial_price]
        
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(0.01, new_price))  # Ensure positive prices
        
        # Generate volumes
        base_volume = 1000
        volumes = np.random.lognormal(np.log(base_volume), 0.5, len(prices))
        
        # Create timestamps
        start_time = datetime.now() - timedelta(minutes=len(prices))
        timestamps = [start_time + timedelta(minutes=i) for i in range(len(prices))]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes.astype(int)
        })
    
    @staticmethod
    def populate_orderbook(orderbook, market_data: pd.DataFrame, depth: int = 5):
        """Populate orderbook with realistic bid/ask levels"""
        if market_data.empty:
            return
        
        current_price = market_data['price'].iloc[-1]
        
        # Determine side enum type
        try:
            if hasattr(orderbook, '__module__') and 'orderbook_engine' in str(type(orderbook)):
                import orderbook_engine
                BUY = orderbook_engine.Side.BUY
                SELL = orderbook_engine.Side.SELL
            else:
                from python_orderbook import Side
                BUY = Side.BUY
                SELL = Side.SELL
        except:
            BUY = "BUY"
            SELL = "SELL"
        
        # Add bid levels (below current price)
        for i in range(depth):
            price = current_price - (i + 1) * 0.01
            quantity = random.randint(100, 500)
            try:
                orderbook.submit_limit_order(BUY, price, quantity)
            except:
                pass
        
        # Add ask levels (above current price)  
        for i in range(depth):
            price = current_price + (i + 1) * 0.01
            quantity = random.randint(100, 500)
            try:
                orderbook.submit_limit_order(SELL, price, quantity)
            except:
                pass


class BacktestingEngine:
    """Main backtesting engine"""
    
    def __init__(self):
        self.results = []
        self.performance_data = {}
    
    def run_strategy_comparison(self, n_orders: int = 50000) -> pd.DataFrame:
        """Run comparison between strategies and engines"""
        print("Starting comprehensive strategy backtesting...")
        
        # Generate market data
        market_data = MarketDataGenerator.generate_price_series(1000)
        
        # Define test strategies
        strategies = [
            VWAPStrategy(target_quantity=10000, target_side="BUY"),
            RandomizedStrategy(target_quantity=10000, target_side="BUY"),
            MovingAverageStrategy(position_size=8000)
        ]
        
        # Test engines
        engines = []
        if CPP_ENGINE_AVAILABLE:
            engines.append(("C++", lambda: orderbook_engine.OrderBook()))
        engines.append(("Python", lambda: PythonOrderBook()))
        
        results = []
        
        for engine_name, engine_factory in engines:
            print(f"\nTesting {engine_name} engine...")
            
            for strategy in strategies:
                print(f"  Running {strategy.name} strategy...")
                
                # Create fresh orderbook
                orderbook = engine_factory()
                
                # Populate with initial market data
                MarketDataGenerator.populate_orderbook(orderbook, market_data)
                
                # Run strategy
                start_time = time.time()
                result = strategy.execute(orderbook, market_data)
                end_time = time.time()
                
                # Add engine info to result
                result_dict = result.to_dict()
                result_dict['engine'] = engine_name
                result_dict['engine_execution_time_ms'] = (end_time - start_time) * 1000
                
                # Add orderbook statistics
                result_dict['total_orders_processed'] = orderbook.get_total_orders()
                result_dict['active_orders_remaining'] = orderbook.get_active_orders_count()
                
                results.append(result_dict)
                print(f"    Completed: {result.trades_count} trades, "
                      f"{result.fill_rate:.1%} fill rate, "
                      f"{result.execution_time_ms:.2f}ms execution time")
        
        # Performance benchmark
        print("\nRunning performance benchmark...")
        benchmark_results = self.run_performance_benchmark(n_orders)
        
        # Combine results
        df_results = pd.DataFrame(results)
        
        # Store for analysis
        self.results = df_results
        self.performance_data = benchmark_results
        
        return df_results
    
    def run_performance_benchmark(self, n_orders: int = 50000) -> Dict[str, float]:
        """Benchmark C++ vs Python engine performance"""
        print(f"Benchmarking with {n_orders:,} orders...")
        
        results = {}
        
        # Generate random orders for benchmark
        np.random.seed(42)
        orders = []
        for i in range(n_orders):
            side = random.choice(['BUY', 'SELL'])
            price = random.uniform(99.0, 101.0)
            quantity = random.randint(10, 1000)
            orders.append((side, price, quantity))
        
        # Benchmark engines
        engines_to_test = []
        if CPP_ENGINE_AVAILABLE:
            import orderbook_engine
            engines_to_test.append(("C++", lambda: orderbook_engine.OrderBook()))
        engines_to_test.append(("Python", lambda: PythonOrderBook()))
        
        for engine_name, engine_factory in engines_to_test:
            print(f"  Benchmarking {engine_name} engine...")
            
            orderbook = engine_factory()
            
            # Determine side enum type
            try:
                if engine_name == "C++" and CPP_ENGINE_AVAILABLE:
                    import orderbook_engine
                    BUY = orderbook_engine.Side.BUY
                    SELL = orderbook_engine.Side.SELL
                else:
                    from python_orderbook import Side
                    BUY = Side.BUY
                    SELL = Side.SELL
            except:
                BUY = "BUY"
                SELL = "SELL"
            
            start_time = time.time()
            
            for side_str, price, quantity in orders:
                side = BUY if side_str == 'BUY' else SELL
                try:
                    orderbook.submit_limit_order(side, price, quantity)
                except Exception as e:
                    # Fallback for different interfaces
                    pass
            
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            
            results[f"{engine_name}_time_ms"] = execution_time
            results[f"{engine_name}_orders_per_second"] = n_orders / (execution_time / 1000)
            results[f"{engine_name}_trades_executed"] = orderbook.get_total_trades()
            
            print(f"    {engine_name}: {execution_time:.2f}ms, "
                  f"{results[f'{engine_name}_orders_per_second']:.0f} orders/sec, "
                  f"{orderbook.get_total_trades()} trades")
        
        # Calculate speedup if both engines available
        if len(engines_to_test) == 2:
            cpp_time = results.get("C++_time_ms", 0)
            python_time = results.get("Python_time_ms", 1)
            if cpp_time > 0 and python_time > 0:
                speedup = python_time / cpp_time
                results["speedup_factor"] = speedup
                print(f"    C++ Speedup: {speedup:.1f}x faster than Python")
        
        return results
    
    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report"""
        if self.results.empty:
            return "No results available. Run backtesting first."
        
        report = []
        report.append("=" * 80)
        report.append("TRADING SYSTEM BACKTESTING REPORT")
        report.append("=" * 80)
        
        # Strategy Performance Summary
        report.append("\n1. STRATEGY PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        for strategy in self.results['strategy_name'].unique():
            strategy_data = self.results[self.results['strategy_name'] == strategy]
            
            report.append(f"\n{strategy} Strategy:")
            report.append(f"  Average Fill Rate: {strategy_data['fill_rate'].mean():.1%}")
            report.append(f"  Average Execution Cost: ${strategy_data['execution_cost'].mean():.2f}")
            report.append(f"  Average Slippage: ${strategy_data['slippage'].mean():.4f}")
            report.append(f"  Total Trades: {strategy_data['trades_count'].sum()}")
            report.append(f"  Volume Traded: {strategy_data['volume_traded'].sum():,.0f}")
        
        # Engine Performance Comparison
        if 'engine' in self.results.columns and len(self.results['engine'].unique()) > 1:
            report.append("\n2. ENGINE PERFORMANCE COMPARISON")
            report.append("-" * 40)
            
            for engine in self.results['engine'].unique():
                engine_data = self.results[self.results['engine'] == engine]
                avg_time = engine_data['engine_execution_time_ms'].mean()
                report.append(f"\n{engine} Engine:")
                report.append(f"  Average Execution Time: {avg_time:.2f}ms")
                report.append(f"  Orders Processed: {engine_data['total_orders_processed'].mean():.0f}")
        
        # Performance Benchmark Results
        if self.performance_data:
            report.append("\n3. PERFORMANCE BENCHMARK RESULTS")
            report.append("-" * 40)
            
            for key, value in self.performance_data.items():
                if 'time_ms' in key:
                    report.append(f"  {key.replace('_', ' ').title()}: {value:.2f}ms")
                elif 'orders_per_second' in key:
                    report.append(f"  {key.replace('_', ' ').title()}: {value:,.0f}")
                elif 'speedup' in key:
                    report.append(f"  {key.replace('_', ' ').title()}: {value:.1f}x")
                elif 'trades' in key:
                    report.append(f"  {key.replace('_', ' ').title()}: {value:,}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


class VisualizationEngine:
    """Generate visualizations for backtesting results"""
    
    @staticmethod
    def create_performance_dashboard(results_df: pd.DataFrame, 
                                   performance_data: Dict[str, float]) -> None:
        """Create comprehensive performance dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Trading System Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Fill Rate Comparison
        ax1 = axes[0, 0]
        fill_rates = results_df.groupby(['strategy_name', 'engine'])['fill_rate'].mean().unstack()
        fill_rates.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Fill Rate by Strategy & Engine')
        ax1.set_ylabel('Fill Rate (%)')
        ax1.set_xlabel('Strategy')
        ax1.legend(title='Engine')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Execution Cost Analysis
        ax2 = axes[0, 1]
        exec_costs = results_df.groupby(['strategy_name', 'engine'])['execution_cost'].mean().unstack()
        exec_costs.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Average Execution Cost')
        ax2.set_ylabel('Execution Cost ($)')
        ax2.set_xlabel('Strategy')
        ax2.legend(title='Engine')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Slippage Distribution
        ax3 = axes[0, 2]
        results_df.boxplot(column='slippage', by='strategy_name', ax=ax3)
        ax3.set_title('Slippage Distribution by Strategy')
        ax3.set_ylabel('Slippage ($)')
        ax3.set_xlabel('Strategy')
        
        # 4. Trade Volume Analysis
        ax4 = axes[1, 0]
        volumes = results_df.groupby(['strategy_name', 'engine'])['volume_traded'].sum().unstack()
        volumes.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_title('Total Volume Traded')
        ax4.set_ylabel('Volume')
        ax4.set_xlabel('Strategy')
        ax4.legend(title='Engine')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Execution Time Comparison
        ax5 = axes[1, 1]
        if 'engine_execution_time_ms' in results_df.columns:
            exec_times = results_df.groupby(['strategy_name', 'engine'])['engine_execution_time_ms'].mean().unstack()
            exec_times.plot(kind='bar', ax=ax5, width=0.8, logy=True)
            ax5.set_title('Execution Time (Log Scale)')
            ax5.set_ylabel('Time (ms)')
            ax5.set_xlabel('Strategy')
            ax5.legend(title='Engine')
            ax5.tick_params(axis='x', rotation=45)
        
        # 6. Performance Benchmark
        ax6 = axes[1, 2]
        if performance_data:
            engines = []
            throughput = []
            for key, value in performance_data.items():
                if 'orders_per_second' in key:
                    engine_name = key.replace('_orders_per_second', '')
                    engines.append(engine_name)
                    throughput.append(value)
            
            if engines:
                bars = ax6.bar(engines, throughput)
                ax6.set_title('Orders Processing Throughput')
                ax6.set_ylabel('Orders/Second')
                ax6.set_xlabel('Engine')
                
                # Add value labels on bars
                for bar, value in zip(bars, throughput):
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughput)*0.01,
                            f'{value:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('trading_system_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def create_strategy_comparison_chart(results_df: pd.DataFrame) -> None:
        """Create detailed strategy comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Strategy Performance Comparison', fontsize=14, fontweight='bold')
        
        # Metrics to compare
        metrics = ['fill_rate', 'execution_cost', 'slippage', 'trades_count']
        titles = ['Fill Rate', 'Execution Cost ($)', 'Slippage ($)', 'Number of Trades']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            # Group by strategy and calculate mean
            data = results_df.groupby('strategy_name')[metric].mean()
            
            bars = ax.bar(data.index, data.values)
            ax.set_title(title)
            ax.set_ylabel(title)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, data.values):
                if metric == 'fill_rate':
                    label = f'{value:.1%}'
                elif 'cost' in metric or 'slippage' in metric:
                    label = f'${value:.3f}'
                else:
                    label = f'{value:.0f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(data.values)*0.01,
                       label, ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main execution function for backtesting"""
    print("Trading System Simulator and Backtester")
    print("=" * 50)
    
    # Initialize backtesting engine
    engine = BacktestingEngine()
    
    # Run comprehensive backtesting
    results_df = engine.run_strategy_comparison(n_orders=25000)
    
    # Generate analysis report
    report = engine.generate_analysis_report()
    print("\n" + report)
    
    # Save results to CSV
    results_df.to_csv('backtesting_results.csv', index=False)
    print(f"\nResults saved to 'backtesting_results.csv'")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        VisualizationEngine.create_performance_dashboard(results_df, engine.performance_data)
        VisualizationEngine.create_strategy_comparison_chart(results_df)
        print("Visualizations saved as PNG files")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    # Print summary statistics for resume
    print("\n" + "="*60)
    print("RESUME-READY SUMMARY STATISTICS")
    print("="*60)
    
    # Calculate key metrics
    total_orders = engine.performance_data.get('Python_orders_per_second', 0) * \
                  (engine.performance_data.get('Python_time_ms', 0) / 1000) if engine.performance_data else 25000
    
    speedup = engine.performance_data.get('speedup_factor', 0)
    total_trades = results_df['trades_count'].sum()
    avg_fill_rate = results_df['fill_rate'].mean()
    total_volume = results_df['volume_traded'].sum()
    
    print(f"• Processed {total_orders:,.0f}+ orders across multiple strategies")
    if speedup > 0:
        print(f"• Achieved {speedup:.1f}x performance improvement with C++/pybind11 optimization")
    print(f"• Executed {total_trades:,} trades with {avg_fill_rate:.1%} average fill rate")
    print(f"• Analyzed ${total_volume:,.0f} in trading volume across VWAP, randomized, and signal-based strategies")
    
    return results_df, engine.performance_data


if __name__ == "__main__":
    main()