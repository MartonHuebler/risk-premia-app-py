"""
Analysis and plotting utilities for the Risk Premia application.
Translated from analysis_utils.R
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def total_returns_plot(returns_df):
    """
    Plot cumulative total returns.
    Translated from analysis_utils.R
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for ticker in returns_df['ticker'].unique():
        ticker_data = returns_df[returns_df['ticker'] == ticker]
        ax.plot(ticker_data['date'], ticker_data['cumreturns'], label=ticker)

    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Cumulative Total Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def rolling_ann_perf(returns_df, window=250):
    """
    Calculate rolling annualized performance statistics.
    Translated from analysis_utils.R
    """
    result = returns_df.groupby('ticker', group_keys=False).apply(
        lambda group: _calc_rolling_perf(group, window)
    )

    # Pivot longer for plotting
    result = result.melt(
        id_vars=['date', 'ticker'],
        value_vars=['roll_ann_return', 'roll_ann_sd', 'roll_sharpe'],
        var_name='metric',
        value_name='value'
    )

    return result


def _calc_rolling_perf(group, window):
    """Calculate rolling performance for a single ticker group."""
    group = group.copy().sort_values('date')

    group['roll_ann_return'] = (
        250 * group['totalreturns'].rolling(window=window, min_periods=window).mean()
    )
    group['roll_ann_sd'] = (
        np.sqrt(250) * group['totalreturns'].rolling(window=window, min_periods=window).std()
    )
    group['roll_sharpe'] = group['roll_ann_return'] / group['roll_ann_sd']

    return group[['date', 'ticker', 'roll_ann_return', 'roll_ann_sd', 'roll_sharpe']]


def rolling_ann_perf_plot(perf_df):
    """
    Plot rolling annualized performance statistics.
    Translated from analysis_utils.R
    """
    metric_names = {
        'roll_ann_return': 'Return',
        'roll_ann_sd': 'Volatility',
        'roll_sharpe': 'Sharpe'
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    for idx, (metric, name) in enumerate(metric_names.items()):
        metric_data = perf_df[perf_df['metric'] == metric]

        for ticker in metric_data['ticker'].unique():
            ticker_data = metric_data[metric_data['ticker'] == ticker]
            axes[idx].plot(ticker_data['date'], ticker_data['value'], label=ticker)

        axes[idx].set_xlabel('Date')
        axes[idx].set_ylabel('Value')
        axes[idx].set_title(name)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle('1-year Rolling Annualised Performance Statistics', fontsize=14, y=1.0)
    fig.tight_layout()

    return fig


def roll_pairwise_corrs(returns_df, period=250):
    """
    Calculate rolling pairwise correlations.
    Translated from analysis_utils.R
    """
    # Create all combinations
    combinations = pd.merge(returns_df, returns_df, on='date', suffixes=('_x', '_y'))

    # Remove diagonal and duplicate pairs
    combinations = combinations[combinations['ticker_x'] != combinations['ticker_y']]

    # Create ticker pair identifier
    def make_pair(row):
        tickers = sorted([row['ticker_x'], row['ticker_y']])
        return f"{tickers[0]}, {tickers[1]}"

    combinations['tickers'] = combinations.apply(make_pair, axis=1)
    combinations = combinations.drop_duplicates(subset=['date', 'tickers'])

    # Calculate rolling correlations
    result = combinations.groupby('tickers', group_keys=False).apply(
        lambda group: _calc_pairwise_corr(group, period)
    )

    return result


def _calc_pairwise_corr(group, period):
    """Calculate rolling correlation for a ticker pair."""
    group = group.copy().sort_values('date')

    correlations = []
    dates = []

    for i in range(period, len(group)):
        window = group.iloc[i-period:i]
        corr = window['totalreturns_x'].corr(window['totalreturns_y'])
        correlations.append(corr)
        dates.append(group.iloc[i]['date'])

    result = pd.DataFrame({
        'date': dates,
        'tickers': group.iloc[0]['tickers'],
        'rollingcor': correlations
    })

    return result


def roll_pairwise_corrs_plot(roll_corr_df, facet=False):
    """
    Plot rolling pairwise correlations.
    Translated from analysis_utils.R
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if facet:
        # Create faceted plot
        ticker_pairs = roll_corr_df['tickers'].unique()
        n_pairs = len(ticker_pairs)
        fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 4*n_pairs))

        if n_pairs == 1:
            axes = [axes]

        for idx, pair in enumerate(ticker_pairs):
            pair_data = roll_corr_df[roll_corr_df['tickers'] == pair]
            axes[idx].plot(pair_data['date'], pair_data['rollingcor'])
            axes[idx].set_title(pair)
            axes[idx].set_xlabel('Date')
            axes[idx].set_ylabel('Correlation')
            axes[idx].grid(True, alpha=0.3)
    else:
        for pair in roll_corr_df['tickers'].unique():
            pair_data = roll_corr_df[roll_corr_df['tickers'] == pair]
            ax.plot(pair_data['date'], pair_data['rollingcor'], label=pair)

        ax.set_xlabel('Date')
        ax.set_ylabel('Correlation')
        ax.set_title('Rolling 12-month Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def cormat(returns_df):
    """
    Calculate correlation matrix.
    Translated from analysis_utils.R
    """
    pivot_df = returns_df.pivot(index='date', columns='ticker', values='totalreturns')
    cor_mat = pivot_df.corr(method='pearson')
    return cor_mat


def cormat_plot(cor_mat):
    """
    Plot correlation matrix as a heatmap.
    Translated from analysis_utils.R
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cor_mat,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax
    )

    ax.set_title('Correlation Matrix')
    fig.tight_layout()

    return fig


def lagged_returns_scatterplot(returns_df, estimation_wdw, forward_wdw, remove_overlapping=True):
    """
    Create scatterplot of lagged returns vs forward returns.
    Translated from analysis_utils.R
    """
    result = returns_df.groupby('ticker', group_keys=False).apply(
        lambda group: _calc_lagged_returns(group, estimation_wdw, forward_wdw, remove_overlapping)
    )

    result = result.dropna()

    # Create faceted plot
    tickers = result['ticker'].unique()
    n_tickers = len(tickers)
    fig, axes = plt.subplots(n_tickers, 1, figsize=(8, 5*n_tickers))

    if n_tickers == 1:
        axes = [axes]

    for idx, ticker in enumerate(tickers):
        ticker_data = result[result['ticker'] == ticker]

        axes[idx].scatter(
            ticker_data['estimation_return'],
            ticker_data['forward_return'],
            alpha=0.6
        )

        # Add regression line
        if len(ticker_data) > 1:
            z = np.polyfit(ticker_data['estimation_return'], ticker_data['forward_return'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(
                ticker_data['estimation_return'].min(),
                ticker_data['estimation_return'].max(),
                100
            )
            axes[idx].plot(x_line, p(x_line), 'r-', alpha=0.8)

        axes[idx].set_title(ticker)
        axes[idx].set_xlabel('Estimation Window Mean Return')
        axes[idx].set_ylabel('Forward Window Mean Return')
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle(
        'Returns vs Forward Returns (annualised)\nAre returns predictive of future returns?',
        fontsize=14,
        y=1.0
    )
    fig.tight_layout()

    return fig


def _calc_lagged_returns(group, estimation_wdw, forward_wdw, remove_overlapping):
    """Calculate lagged returns for a single ticker."""
    group = group.copy().sort_values('date')

    group['estimation_return'] = (
        np.sqrt(250) * group['totalreturns']
        .rolling(window=estimation_wdw).mean()
        .shift(1)
    )

    group['forward_return'] = (
        np.sqrt(250) * group['totalreturns']
        .rolling(window=forward_wdw).mean()
        .shift(-forward_wdw + 1)
    )

    if remove_overlapping:
        step = min(estimation_wdw, forward_wdw)
        indices = group.index[::step]
        group = group.loc[indices]

    return group


def lagged_vol_scatterplot(returns_df, estimation_wdw, forward_wdw, remove_overlapping=True):
    """
    Create scatterplot of lagged volatility vs forward volatility.
    Translated from analysis_utils.R
    """
    result = returns_df.groupby('ticker', group_keys=False).apply(
        lambda group: _calc_lagged_vol(group, estimation_wdw, forward_wdw, remove_overlapping)
    )

    result = result.dropna()

    # Create faceted plot
    tickers = result['ticker'].unique()
    n_tickers = len(tickers)
    fig, axes = plt.subplots(n_tickers, 1, figsize=(8, 5*n_tickers))

    if n_tickers == 1:
        axes = [axes]

    for idx, ticker in enumerate(tickers):
        ticker_data = result[result['ticker'] == ticker]

        axes[idx].scatter(
            ticker_data['estimation_vol'],
            ticker_data['forward_vol'],
            alpha=0.6
        )

        # Add regression line
        if len(ticker_data) > 1:
            z = np.polyfit(ticker_data['estimation_vol'], ticker_data['forward_vol'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(
                ticker_data['estimation_vol'].min(),
                ticker_data['estimation_vol'].max(),
                100
            )
            axes[idx].plot(x_line, p(x_line), 'r-', alpha=0.8)

        axes[idx].set_title(ticker)
        axes[idx].set_xlabel('Estimation Window Volatility')
        axes[idx].set_ylabel('Forward Window Volatility')
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle(
        'Volatility vs Forward Volatility (annualised)\nIs volatility predictive of future volatility?',
        fontsize=14,
        y=1.0
    )
    fig.tight_layout()

    return fig


def _calc_lagged_vol(group, estimation_wdw, forward_wdw, remove_overlapping):
    """Calculate lagged volatility for a single ticker."""
    group = group.copy().sort_values('date')

    group['estimation_vol'] = (
        np.sqrt(250) * group['totalreturns']
        .rolling(window=estimation_wdw).std()
        .shift(1)
    )

    group['forward_vol'] = (
        np.sqrt(250) * group['totalreturns']
        .rolling(window=forward_wdw).std()
        .shift(-forward_wdw + 1)
    )

    if remove_overlapping:
        step = min(estimation_wdw, forward_wdw)
        indices = group.index[::step]
        group = group.loc[indices]

    return group
