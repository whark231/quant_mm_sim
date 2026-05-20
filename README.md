# Quantitative Market-Making Simulator

A from-scratch implementation of a market-making strategy based on the 
Avellaneda-Stoikov (2008) model, applied to live Bitcoin trade data from Coinbase.

## Status
Active development. Core simulation complete. Bayesian optimization and 
regime detection in progress.

## Components
- `src/order_book.py` — limit order book with price-time priority matching
- `src/market_maker.py` — Avellaneda-Stoikov quoting strategy with dynamic risk parameters
- `src/simulator.py` — event-driven simulation loop with OFI adverse selection detection
- `notebooks/` — data exploration, simulation, and performance analysis

## Key Results
- Positive PnL across multiple market samples
- Sharpe ratio improved from 0.041 to 0.054 through sensitivity analysis
- Win rate improved from 36.8% to 46.7% with optimized parameters
