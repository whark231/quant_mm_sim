# Quantitative Market-Making Simulator

## Overview
I built a limit order book, market-making strategy, and market-making simulator from scratch based on the Avellaneda-Stoikov model and applied it to live Bitcoin data. Using iterative analysis and Bayesian optimization with walk-forward validation, I acheived a consistent positive Sharpe ratio across all out-of-sample folds.

## Motivation  
I wanted to know if quant trading/research was something that I was passionate about. This field seems like a black box, and I wanted to gain experience and understanding by attempting to build, learn, and understand fundamental principles and concepts in this space.

## Architecture
- Order Book (`src/order_book.py`) - a limit order book simulator with price-time priority and matching functionality
- Market Maker (`src/market_maker.py`) - a strategy based on the Avellaneda-Stoikov model with dynamic sensitivities 
- Simulator (`src/simulator.py`) - a simulator that runs the Market Maker's strategy on a limit order book filled with live Bitcoin trade data

## Notebooks
- `01_data_exploration.ipynb` — BTC market data analysis and spread estimation
- `02_order_book.ipynb` — order book testing and validation
- `03_simulation.ipynb` — strategy simulation and visualization
- `04_performance_analysis.ipynb` — Sharpe, drawdown, regime analysis, sensitivity analysis
- `05_bayesian_optimization.ipynb` — walk-forward Bayesian optimization
- `06_hmm_training.ipynb` — Hidden Markov Model regime detection, tested in-sample, out-of-sample, and walk-forward

## Strategy
My strategy was to capture positive edge without directional exposure by quoting symmetrically around my reservation price. My reservation price tracked the mid-price of the market. When I'm long my reservation shades under the mid-price to increase the probability of selling and vice versa. My bid-ask spread shifted dynamically based on my inventory, the amount of time remaining in the trading day, and my beliefs about asymmetric information.

## Adverse Selection Detection
In my strategy I tried to detect when there's an imbalance of orders. If one side of my spread is constantly getting filled, it increases my exposure and likely means that other traders are taking advantage of my lack of information. Order flow imbalance captures when this is happening and pauses quoting until there's more symmetry of information.

## Performance Analysis
Early on in the project my strategy suffered and/or was inconsistent due to primarily no parameter tuning or estimation. The strategy didn't perform well in mean-reverting regimes, which is what market makers are supposed to thrive in. My strategy was accidentally directional riding the residual exposure I had at any time step. I had inconsistent and mostly positive Sharpe around 0.1. I would also have large drawdowns when the market moved against my slight directional exposure.

## Bayesian Optimization
After looking at my metrics from univariate analysis, I implemented Bayesian Optimization with Walk-forward validation. Across four folds with different regimes, I maintained positive validation Sharpe with an average Sharpe of 0.157 on out-of-sample data and a mean training Sharpe of 0.152.

## Regime Detection (HMM)
I built a Gaussian Hidden Markov Model from scratch (`src/regime_detector.py`) to classify mean-reverting, trending, and volatile regimes from rolling return, volatility, and order flow imbalance features, then switched my gamma/kappa parameters based on the detected regime. I tested this in-sample, on a single out-of-sample train/validation split, and with the same walk-forward methodology from the Bayesian optimization section.

The regime-conditional strategy consistently cut max drawdown roughly in half versus the static baseline (e.g. $33 vs $62 in-sample, $41 vs $75 out-of-sample), but Sharpe told a mixed story. In walk-forward validation, the static baseline actually edged out the regime-conditional strategy on mean Sharpe (0.059 vs 0.095 across the same four folds), and both fell short of the Bayesian-optimized walk-forward result of 0.157. The HMM doesn't perform much better than the already-optimized baseline — it likely found real regime structure, but my hand-calibrated gamma/kappa per regime weren't tuned the way every parameter in the Bayesian optimization step was, which is the most likely reason it didn't translate into a clear Sharpe improvement.

## Key Findings
- Sigma estimation is important. Volatility is a key measure and it affects the reservation price, optimal spread, and interacts with other important sensitivities.
- Regime detection is important. Different market conditions require different parameters and using the same strategy in different regimes is dangerous.
- Walk-forward validation is essential for financial data. Random train/val splits are unreliable due to non-stationarity since market conditions shift over time and a random split may test on a completely different regime than training.
- How you win and lose is just as important as frequency. Infrequent big wins paired with frequent small losses are good if the wins are big enough. Also, small consistent wins that stack up to cover big losses happen and are okay.
- A good strategy is a predictable and consistent one with low risk.

## Future Work
Folding regime parameters into the Bayesian optimization, rather than hand-calibrating gamma/kappa per regime, would be the natural next step. The HMM seemed to find real regime structure; I'm just not confident I exploited it as well as I could have.

## Data
Performance of the naive strategy was measured with 10,000 BTC trades. For the Bayesian Optimized section of the project 20,000 trades were used. This data was collected from the Coinbase API.

## Requirements
Python 3.10.2, pandas, matplotlib, numpy, scipy, optuna, scikit-learn