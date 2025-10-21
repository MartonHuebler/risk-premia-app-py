# risk-premia-app-py

Simple Risk Premia Strategy App in Python, porting the interactive Shiny application developed by Kris Longmore at RobotWealth.

The strategy is not about market timing, but about balancing long positions of assets that tend to go up in most economic environments. This application allows you to explore buy-and-hold, equal weight rebalancing, and risk parity approaches.

## Original R Application

This is a Python port of the R Shiny application available at: https://github.com/Robot-Wealth/risk-premia-app

## Features

- **Performance Analysis**: View cumulative returns, rolling performance statistics, and correlations
- **Scatterplot Analysis**: Analyze predictability of returns and volatility
- **Backtesting**: Test three different strategies:
  - Equal Weight Buy and Hold
  - Equal Weight Rebalancing
  - Risk Parity (volatility targeting)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/MartonHuebler/risk-premia-app-py.git
cd risk-premia-app-py
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## File Structure

- `app.py` - Main Streamlit application (translated from app.R, analysis_reactives.R, backtest_reactives.R)
- `config.py` - Global configuration constants (translated from global.R)
- `data_loader.py` - Data loading utilities (translated from server_shared.R)
- `analysis_utils.py` - Analysis and plotting functions (translated from analysis_utils.R)
- `backtest_utils.py` - Backtesting functions (translated from backtest_utils.R)
- `data/` - Price data for ETFs and T-bill yields

## Data

The application includes historical price data for:
- US ETFs: VTI, TLT, GLD
- Leveraged US ETFs: UPRO, TMF, UGL
- UCITS ETFs: VDNR.UK, IDTL.UK, IGLN.UK
- 13-week T-bill yields (for margin interest calculations)

## Technologies

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Plotting and visualization
- **pyreadr**: Reading R data files

## License

See LICENSE file for details.
