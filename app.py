"""
Risk Premia Harvesting Streamlit Application
Translated from R Shiny app.R, analysis_reactives.R, and backtest_reactives.R
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import US_ETF_TICKERS, US_LEV_ETF_TICKERS, UCITS_ETF_TICKERS
from data_loader import load_all_data, gg_color_hue
from analysis_utils import (
    total_returns_plot, rolling_ann_perf, rolling_ann_perf_plot,
    roll_pairwise_corrs, roll_pairwise_corrs_plot,
    cormat, cormat_plot,
    lagged_returns_scatterplot, lagged_vol_scatterplot
)
from backtest_utils import (
    num_shares, ew_norebal_positions, get_init_cash_bal,
    bind_cash_positions, adust_bh_backtest_for_margin_calls,
    calc_vol_target, cap_leverage, share_based_backtest,
    summary_performance, stacked_area_chart, trades_chart,
    comm_chart, comm_pct_exp_chart, interest_chart, interest_rate_chart,
    constrained_sizing_plot, combine_port_asset_returns,
    rolling_ann_port_perf, rolling_ann_port_perf_plot
)


# Page configuration
st.set_page_config(
    page_title="Risk Premia Harvesting",
    page_icon="📈",
    layout="wide"
)


# Initialize session state for caching data
@st.cache_data
def get_data():
    """Load all data (cached)."""
    return load_all_data()


# Load data
try:
    data = get_data()
    all_prices = data['all_prices']
    all_monthly_prices = data['all_monthly_prices']
    all_monthly_unadjusted = data['all_monthly_unadjusted']
    monthly_yields = data['monthly_yields']
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# App title
st.title("Risk Premia Harvesting")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Performance", "Scatterplots", "Backtest"])


# ============================================================================
# TAB 1: PERFORMANCE
# ============================================================================
with tab1:
    st.header("Performance Analysis")

    # Sidebar for asset selection
    assets = st.multiselect(
        "Select Assets",
        options=["VTI", "TLT", "GLD"],
        default=["VTI", "TLT", "GLD"],
        key="perf_assets"
    )

    if len(assets) > 0:
        # Filter data
        df = all_prices[all_prices['ticker'].isin(assets)]

        # Cumulative Returns Plot
        st.subheader("Cumulative Total Returns")
        fig = total_returns_plot(df)
        st.pyplot(fig)
        plt.close()

        # Rolling Performance Plot
        st.subheader("1-year Rolling Annualised Performance Statistics")
        perf_data = rolling_ann_perf(df)
        fig = rolling_ann_perf_plot(perf_data)
        st.pyplot(fig)
        plt.close()

        # Rolling Correlation Plot
        st.subheader("Rolling 12-month Correlation")
        corr_data = roll_pairwise_corrs(df)
        fig = roll_pairwise_corrs_plot(corr_data, facet=False)
        st.pyplot(fig)
        plt.close()

        # Correlation Matrix
        st.subheader("Correlation Matrix")
        cor_mat = cormat(df)
        fig = cormat_plot(cor_mat)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Please select at least one asset to display performance metrics.")


# ============================================================================
# TAB 2: SCATTERPLOTS
# ============================================================================
with tab2:
    st.header("Scatterplots - Predictability Analysis")

    col1, col2 = st.columns(2)

    with col1:
        assets_scatter = st.multiselect(
            "Select Assets",
            options=["VTI", "TLT", "GLD"],
            default=["VTI", "TLT", "GLD"],
            key="scatter_assets"
        )

    with col2:
        remove_overlapping = st.checkbox(
            "Show Non-Overlapping Periods Only",
            value=True
        )

    col3, col4 = st.columns(2)

    with col3:
        est_wdw_size = st.slider(
            "Select Estimation Window Length",
            min_value=10,
            max_value=100,
            value=30,
            step=10
        )

    with col4:
        fwd_wdw_size = st.slider(
            "Select Forward Window Length",
            min_value=10,
            max_value=100,
            value=30,
            step=10
        )

    if len(assets_scatter) > 0:
        df_scatter = all_prices[all_prices['ticker'].isin(assets_scatter)]

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Returns vs Forward Returns")
            fig = lagged_returns_scatterplot(
                df_scatter, est_wdw_size, fwd_wdw_size, remove_overlapping
            )
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.subheader("Volatility vs Forward Volatility")
            fig = lagged_vol_scatterplot(
                df_scatter, est_wdw_size, fwd_wdw_size, remove_overlapping
            )
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Please select at least one asset to display scatterplots.")


# ============================================================================
# TAB 3: BACKTEST
# ============================================================================
with tab3:
    st.header("Backtesting")

    # Sidebar for backtest parameters
    st.sidebar.header("Backtest Parameters")

    universe = st.sidebar.selectbox(
        "Select Universe",
        options=[
            ("US ETFs (VTI, TLT, GLD)", "us_etfs"),
            ("Leveraged US ETFs (UPRO, TMF, UGL)", "lev_us_etfs"),
            ("UCITS ETFs (VDNR.UK, IDTL.UK, IGLN.UK)", "ucits_etfs")
        ],
        format_func=lambda x: x[0]
    )
    universe_value = universe[1]

    # Get tickers based on universe
    if universe_value == "us_etfs":
        backtest_tickers = US_ETF_TICKERS
    elif universe_value == "lev_us_etfs":
        backtest_tickers = US_LEV_ETF_TICKERS
    else:
        backtest_tickers = UCITS_ETF_TICKERS

    # Basic parameters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        init_equity = st.slider("Initial Equity, $", 1000, 100000, 20000, 1000)
    with col2:
        max_leverage = st.slider("Maximum Leverage", 0.0, 5.0, 1.0, 0.25)

    # Commission parameters
    col3, col4, col5 = st.sidebar.columns(3)
    with col3:
        comm_per_share = st.slider("Comm. c/share", 0.1, 2.0, 0.5, 0.01)
    with col4:
        min_comm = st.slider("Min Comm. $/order", 0.0, 10.0, 0.5, 0.5)
    with col5:
        margin_interest_spread = st.slider("Int., T-bill spread, %pa", 0.0, 5.0, 2.0, 0.5)

    # Strategy-specific parameters
    same_vol_checkbox = st.sidebar.checkbox("Set asset-specific vol targets", value=False)

    col6, col7, col8 = st.sidebar.columns(3)
    with col6:
        target_vol_1 = st.slider(
            f"{backtest_tickers[0]} Target Volatility, %",
            1.0, 10.0, 5.0, 0.5,
            disabled=False
        )
    with col7:
        target_vol_2 = st.slider(
            f"{backtest_tickers[1]} Target Volatility, %",
            1.0, 10.0, 5.0, 0.5,
            disabled=not same_vol_checkbox
        )
    with col8:
        target_vol_3 = st.slider(
            f"{backtest_tickers[2]} Target Volatility, %",
            1.0, 10.0, 5.0, 0.5,
            disabled=not same_vol_checkbox
        )

    col9, col10, col11 = st.sidebar.columns(3)
    with col9:
        vol_lookback = st.slider("Volatility Estimation Window, days", 5, 120, 60, 5)
    with col10:
        rebal_freq = st.slider("Rebalance Frequency, months", 1, 12, 1, 1)
    with col11:
        cap_freq = st.slider("Frequency to Capitalise Profits", 0, 12, 1, 1)

    run_backtest = st.sidebar.button("UPDATE BACKTEST", type="primary")

    # Create sub-tabs for different strategies
    strategy_tab1, strategy_tab2, strategy_tab3 = st.tabs([
        "Equal Weight Buy and Hold",
        "Equal Weight Rebalance",
        "Risk Parity"
    ])

    # Filter prices for selected universe
    prices = all_prices[all_prices['ticker'].isin(backtest_tickers)]
    monthly_prices = all_monthly_prices[all_monthly_prices['ticker'].isin(backtest_tickers)]
    monthly_unadjusted = all_monthly_unadjusted[all_monthly_unadjusted['ticker'].isin(backtest_tickers)]

    # Calculate start and end dates
    start_date = monthly_prices.groupby('ticker')['date'].min().max()
    end_date = monthly_prices.groupby('ticker')['date'].max().min()

    # Color palette
    app_cols_list = gg_color_hue(5)
    app_cols = dict(zip(list(backtest_tickers) + ['Cash', 'Portfolio'], app_cols_list))

    # Run backtests when button is clicked
    if run_backtest or 'ew_norebal' not in st.session_state:
        with st.spinner("Running backtests..."):
            # Equal Weight Buy & Hold
            shares = num_shares(monthly_prices, init_equity * max_leverage, start_date)
            pos = ew_norebal_positions(
                monthly_prices, shares, comm_per_share/100., min_comm
            )
            init_cash_bal = get_init_cash_bal(pos, init_equity, start_date)
            pos = bind_cash_positions(
                pos, init_cash_bal, init_equity,
                margin_interest_spread, start_date, monthly_yields
            )
            ew_norebal = adust_bh_backtest_for_margin_calls(
                pos, monthly_prices, init_equity, comm_per_share/100.,
                min_comm, margin_interest_spread, monthly_yields
            )
            st.session_state['ew_norebal'] = ew_norebal

            # Equal Weight Rebalance
            ew_rebal = share_based_backtest(
                monthly_prices, monthly_unadjusted, init_equity,
                cap_freq, rebal_freq, comm_per_share/100., min_comm,
                margin_interest_spread, monthly_yields, rebal_method="ew",
                leverage=max_leverage
            )
            st.session_state['ew_rebal'] = ew_rebal

            # Risk Parity
            if same_vol_checkbox:
                vol_targets = {
                    backtest_tickers[0]: target_vol_1 / 100.,
                    backtest_tickers[1]: target_vol_2 / 100.,
                    backtest_tickers[2]: target_vol_3 / 100.
                }
            else:
                vol_targets = target_vol_1 / 100.

            theosize_constrained = calc_vol_target(
                prices, backtest_tickers, vol_lookback, vol_targets
            )
            theosize_constrained = cap_leverage(theosize_constrained, max_leverage=max_leverage)

            volsize_prices = pd.merge(
                monthly_prices,
                theosize_constrained[['ticker', 'date', 'theosize_constrained']],
                on=['ticker', 'date']
            )

            rp_rebal = share_based_backtest(
                volsize_prices, monthly_unadjusted, init_equity,
                cap_freq, rebal_freq, comm_per_share/100., min_comm,
                margin_interest_spread, monthly_yields, rebal_method="rp",
                leverage=max_leverage
            )
            st.session_state['rp_rebal'] = rp_rebal
            st.session_state['volsize_prices'] = volsize_prices

    # Strategy Tab 1: Equal Weight Buy & Hold
    with strategy_tab1:
        if 'ew_norebal' in st.session_state:
            ew_norebal = st.session_state['ew_norebal']

            # Portfolio NAV
            port_nav = ew_norebal.groupby('date')['exposure'].sum().reset_index()
            port_nav['ticker'] = 'NAV'

            # Equity plot
            st.subheader("Portfolio Equity")
            chart_data = pd.concat([
                ew_norebal[['date', 'ticker', 'exposure']],
                port_nav
            ])
            fig = stacked_area_chart(chart_data, 'Equal Weight, No Rebalancing',
                                    backtest_tickers, app_cols)
            st.pyplot(fig)
            plt.close()

            # Performance table
            st.subheader("Performance Summary")
            perf_table = summary_performance(ew_norebal, init_equity, start_date, end_date)
            st.dataframe(perf_table)

            # Rolling performance
            st.subheader("Rolling Performance")
            combined_returns = combine_port_asset_returns(ew_norebal, monthly_prices)
            roll_perf = rolling_ann_port_perf(combined_returns)
            fig = rolling_ann_port_perf_plot(roll_perf, backtest_tickers, app_cols)
            st.pyplot(fig)
            plt.close()

            # Additional charts
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Trades")
                fig = trades_chart(ew_norebal, 'Trades')
                st.pyplot(fig)
                plt.close()

                st.subheader("Commission ($)")
                fig = comm_chart(ew_norebal, 'Commission ($)')
                st.pyplot(fig)
                plt.close()

            with col2:
                st.subheader("Commission as % of Exposure")
                fig = comm_pct_exp_chart(ew_norebal, 'Commission as pct of exposure')
                st.pyplot(fig)
                plt.close()

                st.subheader("Interest ($)")
                fig = interest_chart(ew_norebal, 'Interest ($)')
                st.pyplot(fig)
                plt.close()

            st.subheader("Margin Interest Rate Spread")
            fig = interest_rate_chart(monthly_yields, margin_interest_spread,
                                     'Margin Interest Rate Spread')
            st.pyplot(fig)
            plt.close()

            # Trades table
            st.subheader("Trades Table")
            st.dataframe(ew_norebal.sort_values(['date', 'ticker'], ascending=[False, True]))

    # Strategy Tab 2: Equal Weight Rebalance
    with strategy_tab2:
        if 'ew_rebal' in st.session_state:
            ew_rebal = st.session_state['ew_rebal']

            # Portfolio NAV
            port_nav = ew_rebal.groupby('date')['exposure'].sum().reset_index()
            port_nav['ticker'] = 'NAV'

            # Equity plot
            st.subheader("Portfolio Equity")
            chart_data = pd.concat([
                ew_rebal[['date', 'ticker', 'exposure']],
                port_nav
            ])
            fig = stacked_area_chart(chart_data, 'Equal Weight, Rebalancing',
                                    backtest_tickers, app_cols)
            st.pyplot(fig)
            plt.close()

            # Performance table
            st.subheader("Performance Summary")
            perf_table = summary_performance(ew_rebal, init_equity, start_date, end_date)
            st.dataframe(perf_table)

            # Rolling performance
            st.subheader("Rolling Performance")
            combined_returns = combine_port_asset_returns(ew_rebal, monthly_prices)
            roll_perf = rolling_ann_port_perf(combined_returns)
            fig = rolling_ann_port_perf_plot(roll_perf, backtest_tickers, app_cols)
            st.pyplot(fig)
            plt.close()

            # Additional charts
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Trades")
                fig = trades_chart(ew_rebal, 'Trades')
                st.pyplot(fig)
                plt.close()

                st.subheader("Commission ($)")
                fig = comm_chart(ew_rebal, 'Commission ($)')
                st.pyplot(fig)
                plt.close()

            with col2:
                st.subheader("Commission as % of Exposure")
                fig = comm_pct_exp_chart(ew_rebal, 'Commission as pct of exposure')
                st.pyplot(fig)
                plt.close()

                st.subheader("Interest ($)")
                fig = interest_chart(ew_rebal, 'Interest ($)')
                st.pyplot(fig)
                plt.close()

            st.subheader("Margin Interest Rate Spread")
            fig = interest_rate_chart(monthly_yields, margin_interest_spread,
                                     'Margin Interest Rate Spread')
            st.pyplot(fig)
            plt.close()

            # Trades table
            st.subheader("Trades Table")
            st.dataframe(ew_rebal.sort_values(['date', 'ticker'], ascending=[False, True]))

    # Strategy Tab 3: Risk Parity
    with strategy_tab3:
        if 'rp_rebal' in st.session_state:
            rp_rebal = st.session_state['rp_rebal']
            volsize_prices = st.session_state['volsize_prices']

            # Theoretical sizing plot
            st.subheader("Theoretical Constrained Sizing (% of Portfolio Equity)")
            fig = constrained_sizing_plot(volsize_prices,
                                         'Theoretical Constrained Sizing (% of Portfolio Equity')
            st.pyplot(fig)
            plt.close()

            # Portfolio NAV
            port_nav = rp_rebal.groupby('date')['exposure'].sum().reset_index()
            port_nav['ticker'] = 'NAV'

            # Equity plot
            st.subheader("Portfolio Equity")
            chart_data = pd.concat([
                rp_rebal[['date', 'ticker', 'exposure']],
                port_nav
            ])
            fig = stacked_area_chart(chart_data, 'Risk Parity', backtest_tickers, app_cols)
            st.pyplot(fig)
            plt.close()

            # Performance table
            st.subheader("Performance Summary")
            perf_table = summary_performance(rp_rebal, init_equity, start_date, end_date)
            st.dataframe(perf_table)

            # Rolling performance
            st.subheader("Rolling Performance")
            combined_returns = combine_port_asset_returns(rp_rebal, monthly_prices)
            roll_perf = rolling_ann_port_perf(combined_returns)
            fig = rolling_ann_port_perf_plot(roll_perf, backtest_tickers, app_cols)
            st.pyplot(fig)
            plt.close()

            # Additional charts
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Trades")
                fig = trades_chart(rp_rebal, 'Trades')
                st.pyplot(fig)
                plt.close()

                st.subheader("Commission ($)")
                fig = comm_chart(rp_rebal, 'Commission ($)')
                st.pyplot(fig)
                plt.close()

            with col2:
                st.subheader("Commission as % of Exposure")
                fig = comm_pct_exp_chart(rp_rebal, 'Commission as pct of exposure')
                st.pyplot(fig)
                plt.close()

                st.subheader("Interest ($)")
                fig = interest_chart(rp_rebal, 'Interest ($)')
                st.pyplot(fig)
                plt.close()

            st.subheader("Margin Interest Rate Spread")
            fig = interest_rate_chart(monthly_yields, margin_interest_spread,
                                     'Margin Interest Rate Spread')
            st.pyplot(fig)
            plt.close()

            # Trades table
            st.subheader("Trades Table")
            st.dataframe(rp_rebal.sort_values(['date', 'ticker'], ascending=[False, True]))
