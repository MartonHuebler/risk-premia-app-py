# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python Streamlit web application for backtesting and analyzing risk premia harvesting strategies. It's a port of an R Shiny application originally developed by Kris Longmore at RobotWealth. The application implements three portfolio strategies (equal weight buy-and-hold, equal weight rebalancing, and risk parity) across different ETF universes.

## Development Commands

### Running the Application

**Using uv (recommended):**
```bash
uv run streamlit run app.py
```

**Using pip:**
```bash
streamlit run app.py
```

### Testing

Run the basic component tests:
```bash
python test_app.py
```

### Dependency Management

**With uv:**
```bash
# Sync dependencies from pyproject.toml
uv sync

# Add new dependency
uv add <package-name>
```

**With pip:**
```bash
# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Module Structure

The codebase is organized into focused utility modules that separate concerns:

- **`app.py`**: Main Streamlit application with three tabs (Performance, Scatterplots, Backtest). Contains the UI logic and orchestrates calls to utility functions.
- **`config.py`**: Global constants including ETF ticker lists and margin requirements.
- **`data_loader.py`**: Handles loading price data from RData files and CSV files, calculating returns and cumulative returns.
- **`analysis_utils.py`**: Performance analysis functions including rolling statistics, correlations, and scatterplots.
- **`backtest_utils.py`**: Backtesting engine including position sizing, rebalancing, margin calls, volatility targeting, and performance metrics.

### Data Flow

1. **Data Loading**: `data_loader.py` loads historical price data from `data/` directory (RData format from R) and processes it into standardized DataFrames with date, ticker, close, returns, and cumulative returns columns.

2. **Analysis Tab**: Uses `analysis_utils.py` to compute rolling performance statistics, correlations, and create visualizations from price/return data.

3. **Backtest Tab**: Uses `backtest_utils.py` to:
   - Size positions based on initial capital
   - Track portfolio exposures over time
   - Handle rebalancing trades and commissions
   - Calculate margin requirements and handle margin calls
   - Compute volatility targets for risk parity strategy
   - Generate performance metrics and charts

### Key Data Structures

**Price DataFrames** (returned by `data_loader.py`):
- `all_prices`: Daily prices with returns and cumulative returns
- `all_monthly_prices`: Monthly prices with returns
- `all_monthly_unadjusted`: Monthly prices without adjustments
- `monthly_yields`: T-bill yields for margin interest calculations

All price DataFrames have columns: `date`, `ticker`, `close`, `returns`, `cumreturns`

**Backtest DataFrames** (used in `backtest_utils.py`):
- Positions DataFrame: Contains `date`, `ticker`, `shares`, `close`, `exposure`, `maintenance_margin`, `trades`, `commission`, `interest`, `cash_balance`
- Performance metrics: Returns, Sharpe ratio, max drawdown, CAGR

### Strategy Implementations

1. **Equal Weight Buy & Hold**: Initial equal allocation, no rebalancing
2. **Equal Weight Rebalancing**: Monthly rebalancing to equal weights
3. **Risk Parity**: Volatility targeting with leverage capping, uses rolling volatility estimates to size positions

All strategies account for:
- Transaction costs (per-share commissions with minimums)
- Margin interest on negative cash balances
- Margin calls when maintenance margin is breached

### ETF Universes

The application supports three ETF universes defined in `config.py`:
- **US ETFs**: VTI (stocks), TLT (bonds), GLD (gold)
- **Leveraged US ETFs**: UPRO (3x stocks), TMF (3x bonds), UGL (2x gold)
- **UCITS ETFs**: VDNR.UK, IDTL.UK, IGLN.UK (European equivalents)

### Translation from R

This codebase was translated from R to Python. Key translation patterns:
- R's `dplyr` group_by/summarize → pandas `groupby().apply()`
- R's `ggplot2` → matplotlib/seaborn
- R's `RData` files → loaded via `pyreadr` library
- R's vectorized operations → numpy arrays or pandas operations
- Shiny reactives → Streamlit session state and caching with `@st.cache_data`

### Data Directory

The `data/` directory contains:
- RData files from the original R application
- CSV files for UCITS ETF prices
- Pickle files for Python-serialized results

When adding new data sources, ensure date columns are converted to pandas datetime objects.

## Important Conventions

- All dates are pandas datetime objects
- Returns are simple returns (not log returns)
- Cumulative returns start at 1.0 on the first date
- Rolling windows are specified in trading days (252 per year)
- Volatility is annualized (multiply by sqrt(252))
- All monetary values are in the currency of the ETF (USD for US ETFs)
