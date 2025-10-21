"""
Simple test script to verify the application components work correctly.
"""

import sys

print("Testing Risk Premia Application Components...")
print("=" * 60)

# Test 1: Import all modules
print("\n1. Testing module imports...")
try:
    import config
    import data_loader
    import analysis_utils
    import backtest_utils
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load data
print("\n2. Testing data loading...")
try:
    data = data_loader.load_all_data()
    print(f"   ✓ Loaded {len(data['all_prices'])} price records")
    print(f"   ✓ Found {len(data['all_prices']['ticker'].unique())} unique tickers")
except Exception as e:
    print(f"   ✗ Data loading failed: {e}")
    sys.exit(1)

# Test 3: Test analysis functions
print("\n3. Testing analysis functions...")
try:
    import pandas as pd
    test_df = data['all_prices'][data['all_prices']['ticker'].isin(['VTI', 'TLT', 'GLD'])]

    # Test rolling performance
    perf = analysis_utils.rolling_ann_perf(test_df[:1000])  # Use subset for speed
    print(f"   ✓ Rolling performance calculation successful")

    # Test correlation
    corr = analysis_utils.cormat(test_df[:1000])
    print(f"   ✓ Correlation matrix calculation successful")

except Exception as e:
    print(f"   ✗ Analysis functions failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test backtest functions
print("\n4. Testing backtest functions...")
try:
    monthly_prices = data['all_monthly_prices'][
        data['all_monthly_prices']['ticker'].isin(config.US_ETF_TICKERS)
    ]

    start_date = monthly_prices.groupby('ticker')['date'].min().max()

    # Test number of shares calculation
    shares = backtest_utils.num_shares(monthly_prices, 10000, start_date)
    print(f"   ✓ Position sizing calculation successful")

    # Test vol targeting
    prices_subset = data['all_prices'][
        data['all_prices']['ticker'].isin(config.US_ETF_TICKERS)
    ][:5000]  # Use subset for speed

    vol_targets = backtest_utils.calc_vol_target(
        prices_subset, config.US_ETF_TICKERS, 60, 0.05
    )
    print(f"   ✓ Volatility targeting calculation successful")

except Exception as e:
    print(f"   ✗ Backtest functions failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("\nYou can now run the application with:")
print("  streamlit run app.py")
