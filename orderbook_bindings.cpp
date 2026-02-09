#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include "orderbook.h"

namespace py = pybind11;

PYBIND11_MODULE(orderbook_engine, m) {
    m.doc() = "High-performance C++ Order Book Engine";
    
    // Enums
    py::enum_<Side>(m, "Side")
        .value("BUY", Side::BUY)
        .value("SELL", Side::SELL);
    
    py::enum_<OrderType>(m, "OrderType")
        .value("LIMIT", OrderType::LIMIT)
        .value("MARKET", OrderType::MARKET);
    
    // Order struct
    py::class_<Order>(m, "Order")
        .def_readonly("id", &Order::id)
        .def_readonly("side", &Order::side)
        .def_readonly("type", &Order::type)
        .def_readonly("price", &Order::price)
        .def_readonly("quantity", &Order::quantity)
        .def_readonly("filled_quantity", &Order::filled_quantity)
        .def_readonly("timestamp", &Order::timestamp);
    
    // Trade struct
    py::class_<Trade>(m, "Trade")
        .def_readonly("buy_order_id", &Trade::buy_order_id)
        .def_readonly("sell_order_id", &Trade::sell_order_id)
        .def_readonly("price", &Trade::price)
        .def_readonly("quantity", &Trade::quantity)
        .def_readonly("timestamp", &Trade::timestamp);
    
    // OrderBook class
    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def("submit_limit_order", &OrderBook::submit_limit_order,
             "Submit a limit order",
             py::arg("side"), py::arg("price"), py::arg("quantity"))
        .def("submit_market_order", &OrderBook::submit_market_order,
             "Submit a market order",
             py::arg("side"), py::arg("quantity"))
        .def("cancel_order", &OrderBook::cancel_order,
             "Cancel an order by ID",
             py::arg("order_id"))
        .def("get_best_bid", &OrderBook::get_best_bid,
             "Get the best bid price")
        .def("get_best_ask", &OrderBook::get_best_ask,
             "Get the best ask price")
        .def("get_mid_price", &OrderBook::get_mid_price,
             "Get the mid price")
        .def("get_bid_levels", &OrderBook::get_bid_levels,
             "Get bid levels with depth",
             py::arg("depth") = 5)
        .def("get_ask_levels", &OrderBook::get_ask_levels,
             "Get ask levels with depth",
             py::arg("depth") = 5)
        .def("get_trades", &OrderBook::get_trades,
             "Get all trades", py::return_value_policy::reference_internal)
        .def("get_recent_trades", &OrderBook::get_recent_trades,
             "Get recent trades",
             py::arg("count"))
        .def("get_total_orders", &OrderBook::get_total_orders,
             "Get total orders processed")
        .def("get_total_trades", &OrderBook::get_total_trades,
             "Get total trades executed")
        .def("get_total_volume", &OrderBook::get_total_volume,
             "Get total volume traded")
        .def("get_active_orders_count", &OrderBook::get_active_orders_count,
             "Get number of active orders")
        .def("clear", &OrderBook::clear,
             "Clear all data");
}