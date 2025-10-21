"""
Backtesting utilities for the Risk Premia application.
Translated from backtest_utils.R
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import MAINT_MARGIN


def num_shares(monthlyprices_df, equity, start_date):
    """
    Calculate number of shares to buy for equal weight portfolio.
    Translated from backtest_utils.R
    """
    start_prices = monthlyprices_df[monthlyprices_df['date'] == start_date].copy()
    start_prices['shares'] = np.floor((equity / 3) / start_prices['close']).astype(int)
    return start_prices[['ticker', 'shares']]


def ew_norebal_positions(monthlyprices_df, num_shares_df, per_share_comm, min_comm_per_order):
    """
    Calculate positions, exposures, trades, commissions for equal weight no rebalance strategy.
    Translated from backtest_utils.R
    """
    positions = pd.merge(monthlyprices_df, num_shares_df, on='ticker')

    positions['exposure'] = positions['shares'] * positions['close']
    positions['maintenance_margin'] = MAINT_MARGIN * positions['exposure']

    # Calculate trades
    positions['trades'] = positions.groupby('ticker')['shares'].diff().fillna(positions['shares'])
    positions['tradevalue'] = positions['trades'] * positions['close']

    # Calculate commissions
    positions['commission'] = np.where(
        np.abs(positions['trades']) * per_share_comm > min_comm_per_order,
        np.abs(positions['trades']) * per_share_comm,
        np.where(
            positions['trades'] == 0,
            0,
            min_comm_per_order
        )
    )

    return positions


def get_init_cash_bal(positions, initial_equity, start_date):
    """
    Calculate initial cash balance.
    Translated from backtest_utils.R
    """
    start_positions = positions[positions['date'] == start_date]
    cash = initial_equity - start_positions['exposure'].sum() - start_positions['commission'].sum()
    return cash


def bind_cash_positions(positions, initial_cash_balance, initial_equity, margin_interest_rate,
                       start_date, monthly_yields):
    """
    Add cash positions to the positions dataframe with interest calculations.
    Translated from backtest_utils.R
    """
    # Get dates from positions
    ticker_temp = positions['ticker'].iloc[0]
    dates_df = positions[positions['ticker'] == ticker_temp][['date']].copy()

    # Merge with yields
    cash_df = pd.merge(dates_df, monthly_yields[['date', 'close']], on='date', how='left')
    cash_df = cash_df.rename(columns={'close': 'tbill_rate'})

    # Initialize cash dataframe
    cash_df['ticker'] = 'Cash'
    cash_df['returns'] = 0
    cash_df['close'] = 0
    cash_df['shares'] = 0
    cash_df['exposure'] = 0
    cash_df['tradevalue'] = 0
    cash_df['trades'] = 0
    cash_df['commission'] = 0
    cash_df['maintenance_margin'] = 0
    cash_df['interest'] = 0

    # Loop to calculate interest
    for i in range(len(cash_df)):
        if i == 0:
            cash_df.loc[i, 'exposure'] = initial_cash_balance
            cash_df.loc[i, 'interest'] = 0
            cash_df.loc[i, 'tradevalue'] = initial_cash_balance - initial_equity
        else:
            last_cash = cash_df.loc[i-1, 'exposure']
            this_tbill_rate = cash_df.loc[i, 'tbill_rate']

            if pd.isna(this_tbill_rate):
                this_tbill_rate = 0

            if last_cash < 0:
                interest = last_cash * (margin_interest_rate + this_tbill_rate) / 100 / 12
            else:
                interest = last_cash * max(0, this_tbill_rate - margin_interest_rate) / 100 / 12

            cash_df.loc[i, 'interest'] = interest
            cash_df.loc[i, 'exposure'] = last_cash + interest

    # Negative interest received, positive interest paid
    cash_df['interest'] = -1 * cash_df['interest']

    # Drop tbill_rate and combine with positions
    cash_df = cash_df.drop(columns=['tbill_rate'])
    result = pd.concat([positions, cash_df], ignore_index=True).sort_values('date')

    return result


def bh_margin_call(positions):
    """
    Check for margin calls in buy and hold strategy.
    Translated from backtest_utils.R
    """
    pos_copy = positions.copy()
    pos_copy['is_cash'] = pos_copy['ticker'].apply(lambda x: 'Cash' if x == 'Cash' else 'Stocks')

    margin_calls = (pos_copy.groupby(['date', 'is_cash'])['exposure']
                    .sum()
                    .unstack(fill_value=0)
                    .reset_index())

    if 'Cash' not in margin_calls.columns:
        margin_calls['Cash'] = 0
    if 'Stocks' not in margin_calls.columns:
        margin_calls['Stocks'] = 0

    margin_calls['nav'] = margin_calls['Cash'] + margin_calls['Stocks']
    margin_calls['maintenance_margin'] = MAINT_MARGIN * margin_calls['Stocks']
    margin_calls['margin_call'] = margin_calls['nav'] < margin_calls['maintenance_margin']

    margin_calls['reduce_by'] = np.where(
        margin_calls['nav'] < margin_calls['maintenance_margin'],
        np.minimum(
            np.maximum(0, 1 - (margin_calls['nav'] / MAINT_MARGIN) / margin_calls['Stocks']),
            1
        ),
        0
    )

    return margin_calls


def adust_bh_backtest_for_margin_calls(positions, monthlyprices_df, initial_equity,
                                       per_share_comm, min_comm_per_order,
                                       margin_interest, monthly_yields):
    """
    Adjust buy and hold backtest for margin calls.
    Translated from backtest_utils.R
    """
    margin_calls = bh_margin_call(positions)
    num_margin_calls = margin_calls['margin_call'].sum()

    while num_margin_calls > 0:
        # Find first margin call
        first_margin_call = margin_calls[margin_calls['margin_call']].iloc[0]['date']

        # Keep pre-margin call positions
        no_margin_call_positions = positions[positions['date'] < first_margin_call]

        # Calculate reduced positions
        margin_call_positions = positions[
            (positions['date'] == first_margin_call) &
            (positions['ticker'] != 'Cash')
        ].copy()

        reduce_by = margin_calls[margin_calls['date'] == first_margin_call].iloc[0]['reduce_by']
        margin_call_positions['trades'] = reduce_by * margin_call_positions['shares']
        margin_call_positions['shares'] = margin_call_positions['shares'] * (1 - reduce_by)

        # Calculate cash balance
        if first_margin_call == positions['date'].min():
            starting_cash = initial_equity
            post_margin_call_cash = (
                initial_equity -
                (margin_call_positions['shares'] * margin_call_positions['close']).sum()
            )
        else:
            prev_dates = positions[positions['date'] < first_margin_call]['date'].unique()
            starting_cash_date = prev_dates[-1]
            starting_cash = positions[
                (positions['date'] == starting_cash_date) &
                (positions['ticker'] == 'Cash')
            ]['exposure'].iloc[0]

            post_margin_call_cash = (
                starting_cash -
                (margin_call_positions['trades'] * margin_call_positions['close']).sum()
            )

        # Check if completely wiped out
        if post_margin_call_cash <= 0 and (margin_call_positions['shares'] <= 0).all():
            new_positions = ew_norebal_positions(
                monthlyprices_df[monthlyprices_df['date'] >= first_margin_call],
                margin_call_positions[['ticker', 'shares']],
                per_share_comm,
                min_comm_per_order
            )
            new_positions = bind_cash_positions(
                new_positions, 0, 0, margin_interest, first_margin_call, monthly_yields
            )
        else:
            new_positions = ew_norebal_positions(
                monthlyprices_df[monthlyprices_df['date'] >= first_margin_call],
                margin_call_positions[['ticker', 'shares']],
                per_share_comm,
                min_comm_per_order
            )
            new_positions = bind_cash_positions(
                new_positions, post_margin_call_cash, starting_cash,
                margin_interest, first_margin_call, monthly_yields
            )

        positions = pd.concat([no_margin_call_positions, new_positions], ignore_index=True)
        margin_calls = bh_margin_call(positions)
        num_margin_calls = margin_calls['margin_call'].sum()

    return positions


def calc_vol_target(prices_df, tickers, vol_lookback, target_vol):
    """
    Calculate volatility targeting position sizes.
    Translated from backtest_utils.R
    """
    if isinstance(target_vol, (int, float)):
        # Same vol target for each asset
        result = prices_df.groupby('ticker', group_keys=False).apply(
            lambda group: _calc_vol_target_single(group, vol_lookback, target_vol)
        )
    else:
        # Different vol targets for each asset
        result = prices_df.groupby('ticker', group_keys=False).apply(
            lambda group: _calc_vol_target_multi(group, vol_lookback, target_vol, tickers)
        )

    return result


def _calc_vol_target_single(group, vol_lookback, target_vol):
    """Calculate vol target for single vol target value."""
    group = group.copy().sort_values('date')
    group['returns'] = (group['closeadjusted'] / group['closeadjusted'].shift(1)) - 1
    group['vol'] = group['returns'].rolling(window=vol_lookback, min_periods=vol_lookback).std() * np.sqrt(252)
    group['theosize'] = (target_vol / group['vol']).shift(1)
    return group


def _calc_vol_target_multi(group, vol_lookback, target_vol, tickers):
    """Calculate vol target for multiple vol target values."""
    group = group.copy().sort_values('date')
    ticker = group['ticker'].iloc[0]

    group['returns'] = (group['closeadjusted'] / group['closeadjusted'].shift(1)) - 1
    group['vol'] = group['returns'].rolling(window=vol_lookback, min_periods=vol_lookback).std() * np.sqrt(252)

    if ticker in target_vol:
        group['theosize'] = (target_vol[ticker] / group['vol']).shift(1)
    else:
        group['theosize'] = np.nan

    return group


def cap_leverage(vol_targets, max_leverage=1):
    """
    Enforce leverage constraint on volatility targets.
    Translated from backtest_utils.R
    """
    total_size = (vol_targets.groupby('date')['theosize']
                  .sum()
                  .reset_index()
                  .rename(columns={'theosize': 'totalsize'}))

    total_size['adjfactor'] = np.where(
        total_size['totalsize'] > max_leverage,
        max_leverage / total_size['totalsize'],
        1
    )

    result = pd.merge(vol_targets, total_size, on='date')
    result['theosize_constrained'] = result['theosize'] * result['adjfactor']

    result = result[['ticker', 'date', 'closeadjusted', 'returns', 'vol',
                     'theosize', 'theosize_constrained']].dropna()

    return result


def share_based_backtest(monthlyprices_df, unadjusted_prices, initial_equity,
                        cap_frequency, rebal_frequency, per_share_comm, min_comm,
                        margin_interest_rate, monthly_yields, rebal_method="ew", leverage=1):
    """
    Run share-based backtest for equal weight or risk parity strategy.
    Translated from backtest_utils.R
    """
    assert rebal_method in ["ew", "rp"], "rebal_method must be 'ew' or 'rp'"

    # Create wide dataframes
    wide_prices = monthlyprices_df.pivot(index='date', columns='ticker', values='close')
    wide_unadj_prices = unadjusted_prices.pivot(index='date', columns='ticker', values='close')

    if rebal_method == "rp":
        wide_theosize = monthlyprices_df.pivot(
            index='date', columns='ticker', values='theosize_constrained'
        )

    rowlist = []
    cash = initial_equity
    sharepos = np.zeros(3)
    sharevalue = np.zeros(3)
    equity = initial_equity
    cap_equity = initial_equity

    tickers = wide_prices.columns.tolist()

    for i in range(len(wide_prices)):
        currentdate = wide_prices.index[i]
        currentprice = wide_prices.iloc[i].values
        current_unadj_price = wide_unadj_prices.loc[currentdate].values

        if rebal_method == "rp":
            current_theosize = wide_theosize.iloc[i].values

        # Get T-bill rate
        tbill_row = monthly_yields[monthly_yields['date'] == currentdate]
        if len(tbill_row) > 0:
            this_tbill_rate = tbill_row.iloc[0]['close']
        else:
            this_tbill_rate = monthly_yields.iloc[-1]['close']

        # Calculate interest
        if cash < 0:
            margin_interest = cash * (this_tbill_rate + margin_interest_rate) / 100 / 12
        else:
            margin_interest = cash * max(0, this_tbill_rate - margin_interest_rate) / 100 / 12

        # Update cash and equity
        cash = cash + margin_interest
        equity = np.sum(sharepos * currentprice) + cash

        if equity > 0:
            # Margin call check
            margin_call = False
            liq_shares = np.zeros(3)
            liq_commissions = np.zeros(3)

            if equity < MAINT_MARGIN * np.sum(sharepos * currentprice):
                margin_call = True

                liquidate_factor = 1 - (equity / MAINT_MARGIN) / np.sum(sharepos * currentprice)
                liq_shares = liquidate_factor * sharepos
                liq_commissions = liq_shares / current_unadj_price * per_share_comm
                liq_commissions[liq_commissions < min_comm] = min_comm

                cash = cash - np.sum(liq_shares * currentprice) - np.sum(liq_commissions)
                sharepos = sharepos - liq_shares
                sharevalue = sharepos * currentprice
                equity = np.sum(sharevalue) + cash

            # Update cap equity
            if cap_frequency > 0 and (i % cap_frequency == 0):
                cap_equity = equity

            # Rebalance
            if i == 0 or (i % rebal_frequency == 0):
                if rebal_method == "ew":
                    targetshares = np.floor((leverage * cap_equity / 3) / currentprice).astype(int)
                else:
                    targetshares = np.floor((cap_equity * current_theosize) / currentprice).astype(int)

            trades = targetshares - sharepos
            tradevalue = trades * currentprice
            commissions = np.abs(trades) / current_unadj_price * per_share_comm
            commissions[commissions < min_comm] = np.where(np.abs(trades) > 0, min_comm, 0)

            # Check if we can afford the trades
            post_trade_equity = np.sum(targetshares * currentprice) + cash - np.sum(tradevalue) - np.sum(commissions)

            if post_trade_equity < MAINT_MARGIN * np.sum(targetshares * currentprice):
                max_post_trade_shareval = (equity - np.sum(commissions)) / MAINT_MARGIN

                if max_post_trade_shareval > 0:
                    reduce_by = 1 - max_post_trade_shareval / np.sum(targetshares * currentprice)
                    targetshares = targetshares - np.ceil(reduce_by * targetshares).astype(int)
                    trades = targetshares - sharepos
                    tradevalue = trades * currentprice
                    commissions = np.abs(trades) * per_share_comm
                    commissions[commissions < min_comm] = min_comm

            # Execute trades
            cash = cash - np.sum(tradevalue) - np.sum(commissions)
            sharepos = targetshares
            sharevalue = sharepos * currentprice
            equity = np.sum(sharevalue) + cash

            if equity <= 0:
                # Wiped out
                sharepos = np.zeros(3)
                sharevalue = np.zeros(3)
                trades = np.zeros(3)
                liq_shares = np.zeros(3)
                commissions = np.zeros(3)
                liq_commissions = np.zeros(3)
                equity = 0
                cash = 0
        else:
            # Wiped out
            sharepos = np.zeros(3)
            sharevalue = np.zeros(3)
            tradevalue = np.zeros(3)
            trades = np.zeros(3)
            liq_shares = np.zeros(3)
            commissions = np.zeros(3)
            liq_commissions = np.zeros(3)
            equity = 0
            cash = 0
            margin_call = False

        # Create row dataframe
        for j, ticker in enumerate(['Cash'] + tickers):
            if j == 0:
                row = {
                    'ticker': ticker,
                    'date': currentdate,
                    'close': 0,
                    'shares': 0,
                    'exposure': cash,
                    'sharetrades': 0,
                    'tradevalue': -np.sum(tradevalue),
                    'commission': 0,
                    'interest': -margin_interest,
                    'margin_call': margin_call
                }
            else:
                row = {
                    'ticker': ticker,
                    'date': currentdate,
                    'close': currentprice[j-1],
                    'shares': sharepos[j-1],
                    'exposure': sharevalue[j-1],
                    'sharetrades': trades[j-1] + liq_shares[j-1],
                    'tradevalue': tradevalue[j-1],
                    'commission': commissions[j-1] + liq_commissions[j-1],
                    'interest': 0,
                    'margin_call': margin_call
                }
            rowlist.append(row)

    result = pd.DataFrame(rowlist)
    return result


def calc_port_returns(positions):
    """
    Calculate portfolio returns from positions.
    Translated from backtest_utils.R
    """
    port_returns = (positions.groupby('date')
                    .agg({'exposure': 'sum', 'commission': 'sum'})
                    .reset_index()
                    .rename(columns={'exposure': 'totalequity', 'commission': 'totalcommission'}))

    port_returns = port_returns.sort_values('date')
    port_returns['returns'] = port_returns['totalequity'].pct_change()

    return port_returns


def summary_performance(positions, initial_equity, start_date, end_date):
    """
    Calculate summary performance statistics.
    Translated from backtest_utils.R
    """
    port_returns = calc_port_returns(positions)

    mean_equity = port_returns['totalequity'].mean()
    total_profit = port_returns.iloc[-1]['totalequity'] - initial_equity
    costs_pct_profit = 100 * port_returns['totalcommission'].sum() / total_profit if total_profit != 0 else 0

    # Calculate annualized statistics
    returns_series = port_returns['returns'].dropna()

    if len(returns_series) > 0:
        ann_return = 100 * returns_series.mean() * 12
        ann_vol = 100 * returns_series.std() * np.sqrt(12)
        ann_sharpe = ann_return / ann_vol if ann_vol != 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = 100 * drawdown.min()
    else:
        ann_return = 0
        ann_vol = 0
        ann_sharpe = 0
        max_dd = 0

    # Calculate annual turnover
    total_sell_trades = positions[
        (positions['ticker'] != 'Cash') &
        (positions['tradevalue'] < 0) &
        (positions['date'] != start_date)
    ]['tradevalue'].sum()

    years = (pd.to_datetime(end_date).year - pd.to_datetime(start_date).year)
    if years == 0:
        years = 1

    avg_ann_turnover = -100 * total_sell_trades / (mean_equity * years) if mean_equity != 0 else 0

    result = pd.DataFrame({
        'Ann.Return(%)': [ann_return],
        'Ann.Sharpe(Rf=0%)': [ann_sharpe],
        'Ann.Volatility(%)': [ann_vol],
        'Max.DD(%)': [max_dd],
        'Ave.Ann.Turnover(%)': [avg_ann_turnover],
        'Tot.Profit($)': [total_profit],
        'Costs(%Profit)': [costs_pct_profit]
    })

    return result


# Plotting functions

def stacked_area_chart(positions, title, tickers, colours):
    """
    Create stacked area chart for portfolio exposures.
    Translated from backtest_utils.R
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get non-NAV positions
    non_nav = positions[positions['ticker'] != 'NAV']

    # Pivot for stacking
    pivot_df = non_nav.pivot(index='date', columns='ticker', values='exposure').fillna(0)

    # Ensure Cash and tickers are in the right order
    columns_order = list(tickers) + ['Cash']
    columns_order = [c for c in columns_order if c in pivot_df.columns]
    pivot_df = pivot_df[columns_order]

    # Create stacked area
    pivot_df.plot.area(ax=ax, alpha=0.7)

    # Plot NAV line
    nav_data = positions[positions['ticker'] == 'NAV']
    if len(nav_data) > 0:
        ax.plot(nav_data['date'], nav_data['exposure'], color='black', linewidth=2, label='NAV')

    ax.set_xlabel('Date')
    ax.set_ylabel('Exposure Value')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def trades_chart(positions, title, **kwargs):
    """Plot trade values over time."""
    fig, ax = plt.subplots(figsize=(12, 4))

    non_cash = positions[positions['ticker'] != 'Cash']
    pivot_df = non_cash.pivot(index='date', columns='ticker', values='tradevalue').fillna(0)

    pivot_df.plot.bar(ax=ax, stacked=True, width=10)

    ax.set_xlabel('Date')
    ax.set_ylabel('Trade Value')
    ax.set_title(title)
    ax.legend(loc='best')

    fig.tight_layout()
    return fig


def comm_chart(positions, title, **kwargs):
    """Plot commissions over time."""
    fig, ax = plt.subplots(figsize=(12, 4))

    non_cash = positions[positions['ticker'] != 'Cash']
    pivot_df = non_cash.pivot(index='date', columns='ticker', values='commission').fillna(0)

    pivot_df.plot.bar(ax=ax, stacked=True, width=10)

    ax.set_xlabel('Date')
    ax.set_ylabel('Commission ($)')
    ax.set_title(title)
    ax.legend(loc='best')

    fig.tight_layout()
    return fig


def comm_pct_exp_chart(positions, title, **kwargs):
    """Plot commissions as percentage of exposure."""
    fig, ax = plt.subplots(figsize=(12, 4))

    non_cash = positions[positions['ticker'] != 'Cash'].copy()
    non_cash['commissionpct'] = non_cash['commission'] / non_cash['exposure']

    pivot_df = non_cash.pivot(index='date', columns='ticker', values='commissionpct').fillna(0)

    pivot_df.plot.bar(ax=ax, stacked=True, width=10)

    ax.set_xlabel('Date')
    ax.set_ylabel('Commission (% of Exposure)')
    ax.set_title(title)
    ax.legend(loc='best')

    fig.tight_layout()
    return fig


def interest_chart(positions, title, **kwargs):
    """Plot interest costs/income over time."""
    fig, ax = plt.subplots(figsize=(12, 4))

    cash_only = positions[positions['ticker'] == 'Cash']

    ax.bar(cash_only['date'], cash_only['interest'])

    ax.set_xlabel('Date')
    ax.set_ylabel('Margin Interest Cost')
    ax.set_title(title)
    ax.text(0.05, 0.95, 'Positive interest paid, negative interest received',
            transform=ax.transAxes, va='top', fontsize=10)

    fig.tight_layout()
    return fig


def interest_rate_chart(rates, broker_spread, title):
    """Plot interest rate spread."""
    fig, ax = plt.subplots(figsize=(12, 4))

    rates_df = rates.copy()
    rates_df['accrued_rate'] = np.maximum(0, rates_df['close'] - broker_spread)
    rates_df['deducted_rate'] = rates_df['close'] + broker_spread

    ax.plot(rates_df['date'], rates_df['close'], label='13-week Tbill Yield', color='black')
    ax.plot(rates_df['date'], rates_df['accrued_rate'], label='Accrued Rate',
            color='blue', linestyle='--')
    ax.plot(rates_df['date'], rates_df['deducted_rate'], label='Deducted Rate',
            color='red', linestyle='--')

    ax.set_xlabel('Date')
    ax.set_ylabel('Rate (%)')
    ax.set_title(title)
    ax.legend()
    ax.text(0.05, 0.95,
            'Interest accrues on positive cash balances at lower rate,\nand is debited from negative balances at the higher rate.',
            transform=ax.transAxes, va='top', fontsize=9)

    fig.tight_layout()
    return fig


def constrained_sizing_plot(volsized_prices, title, **kwargs):
    """Plot theoretical constrained sizing over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    pivot_df = volsized_prices.pivot(
        index='date', columns='ticker', values='theosize_constrained'
    ).fillna(0)

    pivot_df.plot.area(ax=ax, alpha=0.7)

    ax.set_xlabel('Date')
    ax.set_ylabel('Constrained Sizing')
    ax.set_title(title)
    ax.legend(loc='best')

    fig.tight_layout()
    return fig


def combine_port_asset_returns(positions, returns_df):
    """Combine portfolio and asset returns."""
    port_returns = calc_port_returns(positions)

    port_returns_formatted = port_returns[['date', 'returns']].copy()
    port_returns_formatted['ticker'] = 'Portfolio'

    asset_returns = returns_df[['ticker', 'date', 'returns']].copy()

    combined = pd.concat([port_returns_formatted, asset_returns], ignore_index=True)
    return combined.sort_values('date')


def rolling_ann_port_perf(port_returns_df, window=24):
    """Calculate rolling annualized portfolio performance."""
    result = port_returns_df.groupby('ticker', group_keys=False).apply(
        lambda group: _calc_rolling_port_perf(group, window)
    )

    result = result.melt(
        id_vars=['date', 'ticker'],
        value_vars=['roll_ann_return', 'roll_ann_sd', 'roll_sharpe'],
        var_name='metric',
        value_name='value'
    )

    return result


def _calc_rolling_port_perf(group, window):
    """Calculate rolling performance for monthly returns."""
    group = group.copy().sort_values('date')

    group['roll_ann_return'] = 12 * group['returns'].rolling(window=window, min_periods=window).mean()
    group['roll_ann_sd'] = np.sqrt(12) * group['returns'].rolling(window=window, min_periods=window).std()
    group['roll_sharpe'] = group['roll_ann_return'] / group['roll_ann_sd']

    return group[['date', 'ticker', 'roll_ann_return', 'roll_ann_sd', 'roll_sharpe']]


def rolling_ann_port_perf_plot(perf_df, tickers, ticker_colours, **kwargs):
    """Plot rolling annualized portfolio performance."""
    metric_names = {
        'roll_ann_return': 'Return',
        'roll_ann_sd': 'Volatility',
        'roll_sharpe': 'Sharpe'
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for idx, (metric, name) in enumerate(metric_names.items()):
        metric_data = perf_df[perf_df['metric'] == metric]

        for ticker in metric_data['ticker'].unique():
            ticker_data = metric_data[metric_data['ticker'] == ticker]
            linewidth = 2 if ticker == 'Portfolio' else 1
            axes[idx].plot(ticker_data['date'], ticker_data['value'],
                          label=ticker, linewidth=linewidth)

        axes[idx].set_xlabel('Date')
        axes[idx].set_ylabel('Value')
        axes[idx].set_title(name)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle('2-year Rolling Annualised Performance Statistics', fontsize=14)
    fig.tight_layout()

    return fig
