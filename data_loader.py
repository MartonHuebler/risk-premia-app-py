"""
Data loading functions for the Risk Premia application.
Translated from server_shared.R
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyreadr
from config import US_ETF_TICKERS, US_LEV_ETF_TICKERS, UCITS_ETF_TICKERS

# Base data directory
DATA_DIR = Path(__file__).parent / "data"


def load_rdata(filename):
    """Load an RData file and return the first dataframe."""
    result = pyreadr.read_r(str(DATA_DIR / filename))
    # RData files can contain multiple objects, get the first one
    return list(result.values())[0]


def load_us_etf_prices():
    """Load US ETF prices from RData file."""
    df = load_rdata("us_etf_prices.RData")
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_us_lev_etf_prices():
    """Load leveraged US ETF prices from RData file."""
    df = load_rdata("us_lev_etf_prices.RData")
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_ucits_etf_prices():
    """Load UCITS ETF prices from RData or CSV files."""
    # Try RData first, fall back to CSV
    try:
        df = load_rdata("ucits_etf_prices.RData")
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        # Load from CSV files
        csv_files = [
            "VDNR-UK.csv",
            "IDTL-UK.csv",
            "IGLN.UK.csv"
        ]
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(DATA_DIR / csv_file)
            # Convert date from DD/MM/YYYY format
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True).sort_values('date')


def load_tbill_yields():
    """Load T-bill yield data from RData file."""
    df = load_rdata("tbill_yields.RData")
    df['date'] = pd.to_datetime(df['date'])
    return df


def add_total_returns_col(prices_df):
    """
    Add total returns and cumulative returns columns to price dataframe.
    Translated from analysis_utils.R
    """
    prices_df = prices_df.sort_values('date')

    # Group by ticker and calculate returns
    result = prices_df.groupby('ticker', group_keys=False).apply(
        _calculate_returns
    )

    return result.dropna()


def _calculate_returns(group):
    """Calculate returns for a single ticker group."""
    group = group.copy()
    group['totalreturns'] = (group['closeadjusted'] / group['closeadjusted'].shift(1)) - 1
    group['dividends'] = group['dividends'].fillna(0)
    group['volume'] = group['volume'].fillna(0)
    group['cumreturns'] = (1 + group['totalreturns']).cumprod()
    return group


def get_monthends(prices_df, ticker_var='ticker'):
    """
    Get month-end dates from price dataframe.
    Translated from backtest_utils.R
    """
    df = prices_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Use ticker or symbol column
    if ticker_var not in df.columns:
        if 'ticker' in df.columns:
            ticker_var = 'ticker'
        elif 'symbol' in df.columns:
            ticker_var = 'symbol'

    monthends = (df.groupby(['year', 'month', ticker_var])['date']
                 .max()
                 .reset_index())

    return monthends


def make_monthly_prices(prices_df, monthends, ticker_var='ticker', price_var='closeadjusted'):
    """
    Create monthly price dataframe from daily prices.
    Translated from backtest_utils.R
    """
    if 'symbol' in prices_df.columns:
        merged = pd.merge(
            prices_df,
            monthends,
            left_on=['date', 'symbol'],
            right_on=['date', 'symbol']
        )
        result = merged[[ticker_var, 'date', price_var]].rename(columns={price_var: 'close'})
    else:
        merged = pd.merge(
            prices_df,
            monthends,
            left_on=['date', 'ticker'],
            right_on=['date', 'ticker']
        )
        result = merged[[ticker_var, 'date', price_var]].rename(columns={price_var: 'close'})

    return result


def gg_color_hue(n):
    """
    Generate n colors using ggplot2's default color palette.
    Translated from server_shared.R
    """
    hues = np.linspace(15, 375, n + 1)[:n]
    # Convert HCL to RGB (approximation)
    colors = []
    for h in hues:
        # Simple conversion for visualization
        colors.append(f'hsl({h}, 100%, 65%)')
    return colors


def load_all_data():
    """
    Load and prepare all data for the application.
    Returns a dictionary with all loaded dataframes.
    """
    # Load price data
    us_etf_prices = load_us_etf_prices()
    us_lev_etf_prices = load_us_lev_etf_prices()
    ucits_etf_prices = load_ucits_etf_prices()
    tbill_yields = load_tbill_yields()

    # Combine all prices
    all_prices = pd.concat([
        us_etf_prices,
        us_lev_etf_prices,
        ucits_etf_prices
    ], ignore_index=True).sort_values('date')

    # Add total returns
    all_prices = add_total_returns_col(all_prices)

    # Set up monthly prices and returns
    month_ends = get_monthends(all_prices)

    all_monthly_prices = make_monthly_prices(all_prices, month_ends, ticker_var='ticker')
    all_monthly_prices = all_monthly_prices.sort_values('date')
    all_monthly_prices['returns'] = (
        all_monthly_prices.groupby('ticker')['close']
        .pct_change()
    )

    # Monthly unadjusted prices
    all_monthly_unadjusted = make_monthly_prices(
        all_prices, month_ends, price_var='close'
    ).sort_values('date')

    # Set up monthly t-bill yields
    irx_monthends = get_monthends(tbill_yields, ticker_var='symbol')
    monthly_yields = make_monthly_prices(
        tbill_yields, irx_monthends, ticker_var='symbol', price_var='series'
    ).sort_values('date')

    # Ensure prices and yields have common index
    ticker_temp = all_monthly_prices['ticker'].iloc[0]
    common_dates = all_monthly_prices[all_monthly_prices['ticker'] == ticker_temp][['date', 'ticker']]

    monthly_yields = pd.merge(
        common_dates[['date']],
        monthly_yields,
        on='date',
        how='left'
    )
    monthly_yields['symbol'] = monthly_yields['symbol'].ffill()
    monthly_yields['close'] = monthly_yields['close'].ffill()

    return {
        'all_prices': all_prices,
        'us_etf_prices': us_etf_prices,
        'us_lev_etf_prices': us_lev_etf_prices,
        'ucits_etf_prices': ucits_etf_prices,
        'tbill_yields': tbill_yields,
        'all_monthly_prices': all_monthly_prices,
        'all_monthly_unadjusted': all_monthly_unadjusted,
        'monthly_yields': monthly_yields
    }
