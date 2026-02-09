#ifndef ORDERBOOK_H
#define ORDERBOOK_H

#include <map>
#include <queue>
#include <vector>
#include <memory>
#include <string>
#include <chrono>

enum class Side {
    BUY,
    SELL
};

enum class OrderType {
    LIMIT,
    MARKET
};

struct Order {
    uint64_t id;
    Side side;
    OrderType type;
    double price;
    uint64_t quantity;
    uint64_t filled_quantity;
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    
    Order(uint64_t id, Side side, OrderType type, double price, uint64_t quantity)
        : id(id), side(side), type(type), price(price), quantity(quantity), 
          filled_quantity(0), timestamp(std::chrono::high_resolution_clock::now()) {}
};

struct Trade {
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    double price;
    uint64_t quantity;
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    
    Trade(uint64_t buy_id, uint64_t sell_id, double price, uint64_t quantity)
        : buy_order_id(buy_id), sell_order_id(sell_id), price(price), 
          quantity(quantity), timestamp(std::chrono::high_resolution_clock::now()) {}
};

// Price level containing orders at the same price (FIFO queue)
class PriceLevel {
public:
    double price;
    std::queue<std::shared_ptr<Order>> orders;
    uint64_t total_quantity;
    
    PriceLevel(double p) : price(p), total_quantity(0) {}
    
    void add_order(std::shared_ptr<Order> order) {
        orders.push(order);
        total_quantity += (order->quantity - order->filled_quantity);
    }
    
    std::shared_ptr<Order> get_front_order() {
        return orders.empty() ? nullptr : orders.front();
    }
    
    void remove_front_order() {
        if (!orders.empty()) {
            auto order = orders.front();
            total_quantity -= (order->quantity - order->filled_quantity);
            orders.pop();
        }
    }
    
    bool is_empty() const {
        return orders.empty();
    }
};

class OrderBook {
private:
    // Buy orders: higher prices first (descending order)
    std::map<double, std::shared_ptr<PriceLevel>, std::greater<double>> buy_levels;
    // Sell orders: lower prices first (ascending order)
    std::map<double, std::shared_ptr<PriceLevel>, std::less<double>> sell_levels;
    
    std::map<uint64_t, std::shared_ptr<Order>> active_orders;
    std::vector<Trade> trades;
    uint64_t next_order_id;
    
    // Statistics
    uint64_t total_orders_processed;
    uint64_t total_trades_executed;
    double total_volume_traded;
    
public:
    OrderBook() : next_order_id(1), total_orders_processed(0), 
                  total_trades_executed(0), total_volume_traded(0.0) {}
    
    // Submit a limit order
    uint64_t submit_limit_order(Side side, double price, uint64_t quantity);
    
    // Submit a market order
    uint64_t submit_market_order(Side side, uint64_t quantity);
    
    // Cancel an order
    bool cancel_order(uint64_t order_id);
    
    // Get best bid/ask prices
    double get_best_bid() const;
    double get_best_ask() const;
    double get_mid_price() const;
    
    // Get market depth
    std::vector<std::pair<double, uint64_t>> get_bid_levels(int depth = 5) const;
    std::vector<std::pair<double, uint64_t>> get_ask_levels(int depth = 5) const;
    
    // Get trades
    const std::vector<Trade>& get_trades() const { return trades; }
    std::vector<Trade> get_recent_trades(size_t count) const;
    
    // Statistics
    uint64_t get_total_orders() const { return total_orders_processed; }
    uint64_t get_total_trades() const { return total_trades_executed; }
    double get_total_volume() const { return total_volume_traded; }
    size_t get_active_orders_count() const { return active_orders.size(); }
    
    // Clear all data (for benchmarking)
    void clear();
    
private:
    void match_order(std::shared_ptr<Order> order);
    void execute_trade(std::shared_ptr<Order> buy_order, std::shared_ptr<Order> sell_order, 
                      double price, uint64_t quantity);
};

// Implementation

uint64_t OrderBook::submit_limit_order(Side side, double price, uint64_t quantity) {
    auto order = std::make_shared<Order>(next_order_id++, side, OrderType::LIMIT, price, quantity);
    active_orders[order->id] = order;
    total_orders_processed++;

    // Try to match the order first
    match_order(order);

    // If order not fully filled, add it to the order book
    if (order->filled_quantity < order->quantity) {
        if (side == Side::BUY) {
            if (buy_levels.find(price) == buy_levels.end()) {
                buy_levels[price] = std::make_shared<PriceLevel>(price);
            }
            buy_levels[price]->add_order(order);
        } else { // Side::SELL
            if (sell_levels.find(price) == sell_levels.end()) {
                sell_levels[price] = std::make_shared<PriceLevel>(price);
            }
            sell_levels[price]->add_order(order);
        }
    } else {
        // Order fully filled, remove from active orders
        active_orders.erase(order->id);
    }

    return order->id;
}


uint64_t OrderBook::submit_market_order(Side side, uint64_t quantity) {
    auto order = std::make_shared<Order>(next_order_id++, side, OrderType::MARKET, 0.0, quantity);
    total_orders_processed++;
    
    match_order(order);
    
    // Market orders are never added to the book
    return order->id;
}

bool OrderBook::cancel_order(uint64_t order_id) {
    auto it = active_orders.find(order_id);
    if (it == active_orders.end()) {
        return false; // Order not found
    }
    
    auto order = it->second;
    active_orders.erase(it);
    
    // Remove from price level
    if (order->side == Side::BUY) {
        auto level_it = buy_levels.find(order->price);
        if (level_it != buy_levels.end()) {
            auto& level = level_it->second;
            level->total_quantity -= (order->quantity - order->filled_quantity);
            if (level->total_quantity == 0) {
                buy_levels.erase(level_it);
            }
        }
    } else { // Side::SELL
        auto level_it = sell_levels.find(order->price);
        if (level_it != sell_levels.end()) {
            auto& level = level_it->second;
            level->total_quantity -= (order->quantity - order->filled_quantity);
            if (level->total_quantity == 0) {
                sell_levels.erase(level_it);
            }
        }
    }
    
    return true;
}


void OrderBook::match_order(std::shared_ptr<Order> order) {
    if (order->side == Side::BUY) {
        // BUY orders match against sell_levels
        while (order->filled_quantity < order->quantity && !sell_levels.empty()) {
            auto best_level_it = sell_levels.begin();
            auto best_level = best_level_it->second;

            // Check if we can match
            bool can_match = (order->type == OrderType::MARKET) || (order->price >= best_level->price);
            if (!can_match) break;

            // Match with orders in this level
            while (!best_level->is_empty() && order->filled_quantity < order->quantity) {
                auto matching_order = best_level->get_front_order();

                // Skip cancelled orders
                if (active_orders.find(matching_order->id) == active_orders.end()) {
                    best_level->remove_front_order();
                    continue;
                }

                uint64_t trade_quantity = std::min(
                    order->quantity - order->filled_quantity,
                    matching_order->quantity - matching_order->filled_quantity
                );

                execute_trade(order, matching_order, best_level->price, trade_quantity);

                order->filled_quantity += trade_quantity;
                matching_order->filled_quantity += trade_quantity;

                // Remove fully filled order
                if (matching_order->filled_quantity == matching_order->quantity) {
                    active_orders.erase(matching_order->id);
                    best_level->remove_front_order();
                }
            }

            // Remove empty level
            if (best_level->is_empty()) {
                sell_levels.erase(best_level_it);
            }
        }
    } else { // Side::SELL
        // SELL orders match against buy_levels
        while (order->filled_quantity < order->quantity && !buy_levels.empty()) {
            auto best_level_it = buy_levels.begin();
            auto best_level = best_level_it->second;

            // Check if we can match
            bool can_match = (order->type == OrderType::MARKET) || (order->price <= best_level->price);
            if (!can_match) break;

            // Match with orders in this level
            while (!best_level->is_empty() && order->filled_quantity < order->quantity) {
                auto matching_order = best_level->get_front_order();

                // Skip cancelled orders
                if (active_orders.find(matching_order->id) == active_orders.end()) {
                    best_level->remove_front_order();
                    continue;
                }

                uint64_t trade_quantity = std::min(
                    order->quantity - order->filled_quantity,
                    matching_order->quantity - matching_order->filled_quantity
                );

                execute_trade(matching_order, order, best_level->price, trade_quantity);

                order->filled_quantity += trade_quantity;
                matching_order->filled_quantity += trade_quantity;

                // Remove fully filled order
                if (matching_order->filled_quantity == matching_order->quantity) {
                    active_orders.erase(matching_order->id);
                    best_level->remove_front_order();
                }
            }

            // Remove empty level
            if (best_level->is_empty()) {
                buy_levels.erase(best_level_it);
            }
        }
    }
}


void OrderBook::execute_trade(std::shared_ptr<Order> buy_order, std::shared_ptr<Order> sell_order, 
                             double price, uint64_t quantity) {
    trades.emplace_back(buy_order->id, sell_order->id, price, quantity);
    total_trades_executed++;
    total_volume_traded += price * quantity;
}

double OrderBook::get_best_bid() const {
    return buy_levels.empty() ? 0.0 : buy_levels.begin()->first;
}

double OrderBook::get_best_ask() const {
    return sell_levels.empty() ? 0.0 : sell_levels.begin()->first;
}

double OrderBook::get_mid_price() const {
    double bid = get_best_bid();
    double ask = get_best_ask();
    if (bid > 0 && ask > 0) {
        return (bid + ask) / 2.0;
    }
    return 0.0;
}

std::vector<std::pair<double, uint64_t>> OrderBook::get_bid_levels(int depth) const {
    std::vector<std::pair<double, uint64_t>> levels;
    int count = 0;
    for (const auto& level : buy_levels) {
        if (count >= depth) break;
        levels.emplace_back(level.first, level.second->total_quantity);
        count++;
    }
    return levels;
}

std::vector<std::pair<double, uint64_t>> OrderBook::get_ask_levels(int depth) const {
    std::vector<std::pair<double, uint64_t>> levels;
    int count = 0;
    for (const auto& level : sell_levels) {
        if (count >= depth) break;
        levels.emplace_back(level.first, level.second->total_quantity);
        count++;
    }
    return levels;
}

std::vector<Trade> OrderBook::get_recent_trades(size_t count) const {
    if (trades.size() <= count) {
        return trades;
    }
    return std::vector<Trade>(trades.end() - count, trades.end());
}

void OrderBook::clear() {
    buy_levels.clear();
    sell_levels.clear();
    active_orders.clear();
    trades.clear();
    next_order_id = 1;
    total_orders_processed = 0;
    total_trades_executed = 0;
    total_volume_traded = 0.0;
}

#endif  