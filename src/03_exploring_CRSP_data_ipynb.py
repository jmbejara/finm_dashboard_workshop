# %%
"""
# Exploring CRSP Data with Python

This script demonstrates how to explore and analyze CRSP daily stock data using various Python plotting libraries.

## Overview
We'll analyze CRSP daily stock data for selected stocks (AAPL, JNJ, TSLA) and compare their performance against the S&P 500 index. The analysis includes:
- Cumulative returns comparison
- Dividend analysis
- Rolling volatility analysis
- Multiple plotting approaches (Matplotlib, Seaborn, Plotly)

## Data Source
- CRSP Daily Stock File v2 (DSF) via WRDS
- Filtered for common stock universe with proper exchange and trading filters
- Date range: 2019-2025
"""

# %%
import pandas as pd
import numpy as np
import wrds
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

import config

DATA_DIR = config.DATA_DIR
WRDS_USERNAME = config.WRDS_USERNAME

db = wrds.Connection(wrds_username=WRDS_USERNAME)

# %%
"""
## Understanding CRSP Data Structure

To find the right table, we use a combination of the web query interface and the SAS Studio explorer.
The web query interface is available at:
https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/annual-update/stock-version-2/daily-stock-file/

Note: Web queries often use merges of tables from many different sources. The results of these merges are not usually available through the Python API interface. Often, you'll have to merge the tables yourself.

However, in this case, we can use the SAS Studio explorer to find the right table. Lucky for us, the data in the web query is available in a pre-merged table available through the Python API.
"""

# %%
# First, let's look at the standard daily stock file (DSF) from the CIZ format
dsf = db.get_table(library="crsp", table="dsf_v2", obs=10)
dsf.head()

# %%
dsf.info()

# %%
"""
## Finding the Pre-merged Table

Now, let's find the pre-merged table that contains the data we want.
Notice that it corresponds to the web query we used above.
"""

# %%
df = db.get_table(library="crsp", table="wrds_dsfv2_query", obs=10)
df.head()

# %%
"""
Note: We actually just made a mistake above. For some reason, we aren't allowed to access this via "crspa", but we are via "crsp".
"""

# %%
df.info()

# %%
"""
Notice that this now matches the web query variables list.
"""

# %%
"""
## Defining the Data Query

Now, let's explore some of the columns. But first, we need to download more data.
"""

# %%
query = """
SELECT 
    permno, 
    permco, 
    dlycaldt, 
    issuertype, 
    securitytype, 
    securitysubtype, 
    sharetype, 
    usincflg, 
    primaryexch, 
    conditionaltype, 
    tradingstatusflg,
    dlyret, 
    dlyretx, 
    dlyreti,
    dlyorddivamt,
    dlynonorddivamt,
    shrout, 
    dlyprc,
    ticker,
    securitynm,
    sprtrn,
    vwretd
FROM 
    crsp.wrds_dsfv2_query
WHERE 
    dlycaldt between '01/01/2019' and '01/01/2025' AND
    sharetype = 'NS' AND
    securitytype = 'EQTY' AND
    securitysubtype = 'COM' AND
    usincflg = 'Y' AND
    issuertype IN ('ACOR', 'CORP') AND
    primaryexch IN ('N', 'A', 'Q') AND
    conditionaltype = 'RW' AND
    tradingstatusflg = 'A'
"""

# %%
def pull_crsp_sample(data_dir=DATA_DIR):
    """
    Pull CRSP daily stock data with comprehensive filtering for common stock universe.
    
    This function implements the equivalent of legacy CRSP filters:
    - shrcd = 10 or 11 (common stock)
    - exchcd = 1, 2, or 3 (NYSE, AMEX, NASDAQ)
    
    Filters applied:
    1. Date range: 2019-2025
    2. Common stock universe:
       - sharetype = 'NS' (New Shares)
       - securitytype = 'EQTY' (Equity)
       - securitysubtype = 'COM' (Common Stock)
       - usincflg = 'Y' (US Incorporated)
       - issuertype IN ('ACOR', 'CORP') (Accordion or Corporate)
    3. Exchange and trading filters:
       - primaryexch IN ('N', 'A', 'Q') (NYSE, AMEX, NASDAQ)
       - conditionaltype = 'RW' (Regular Way trading)
       - tradingstatusflg = 'A' (Active trading status)
    
    Caching:
    - Data is cached locally as a parquet file to avoid repeated WRDS queries
    - If cached data exists, it loads from disk instead of querying WRDS
    - If no cache exists, queries WRDS and saves the result for future use
    
    Args:
        data_dir: Directory to store/load cached data
        
    Returns:
        DataFrame: Filtered CRSP daily stock data
    """
    data_path = data_dir / "crsp_dsf_v2_example.parquet"
    if data_path.exists():
        df = pd.read_parquet(data_path)
    else:
        df = db.raw_sql(query, date_cols=["dlycaldt"])
        df.to_parquet(data_path)
    return df

# %%
df = pull_crsp_sample()
df.head()

# %%
df.info()

# %%
"""
## Data Summary and Exploration

Let's look at some summary statistics to understand our dataset.
"""

# %%
# Let's look at some summary statistics
print("\n=== Data Summary ===")
print(f"Date range: {df['dlycaldt'].min()} to {df['dlycaldt'].max()}")
print(f"Number of unique stocks: {df['permno'].nunique()}")
print(f"Total observations: {len(df)}")

# %%
"""
## Identifying Well-Known Stocks

We'll look for stocks with recognizable tickers and good data coverage for our analysis.
"""

# %%
# Let's identify some well-known stocks for analysis
# We'll look for stocks with recognizable tickers and good data coverage
stock_summary = df.groupby(['permno', 'ticker', 'securitynm']).agg({
    'dlycaldt': ['count', 'min', 'max'],
    'dlyret': 'count',
    'dlyorddivamt': lambda x: (x > 0).sum()
}).round(2)

stock_summary.columns = ['obs_count', 'start_date', 'end_date', 'return_obs', 'dividend_days']
stock_summary = stock_summary.reset_index()

# Filter for stocks with good data coverage and recognizable names
good_stocks = stock_summary[
    (stock_summary['obs_count'] > 500) &  # At least 500 observations
    (stock_summary['ticker'].notna()) &   # Has a ticker
    (stock_summary['ticker'] != '')       # Ticker is not empty
].sort_values('obs_count', ascending=False)

print("\n=== Top Stocks by Data Coverage ===")
good_stocks.head(20)

# %%
"""
## Selecting Stocks for Analysis

Let's select three stocks for analysis:
1. A dividend-paying stock (likely a utility or financial)
2. A growth stock that pays no dividends (likely tech)
3. A stock in between
"""

# %%
# Let's look for some specific well-known stocks
target_tickers = ['AAPL', 'MSFT', 'JNJ', 'PG', 'KO', 'XOM', 'JPM', 'WMT', 'NVDA', 'TSLA']
available_stocks = good_stocks[good_stocks['ticker'].isin(target_tickers)]

print("\n=== Available Target Stocks ===")
available_stocks

# %%
# Select three stocks for analysis
selected_stocks = ['AAPL', 'JNJ', 'TSLA']  # Apple (tech, some dividends), J&J (dividend payer), Tesla (no dividends)

# Filter data for selected stocks and market
selected_data = df[df['ticker'].isin(selected_stocks)].copy()
market_data = df[['dlycaldt', 'sprtrn', 'vwretd']].drop_duplicates().copy()

# %%
"""
## Data Quality Check

Let's verify our data quality and check for any duplicate entries.
"""

# %%
# Debug: Check for duplicate dates in selected data
print(f"\n=== Data Quality Check ===")
print(f"Selected stocks data shape: {selected_data.shape}")
print(f"Duplicate ticker-date combinations: {selected_data.duplicated(subset=['ticker', 'dlycaldt']).sum()}")
print(f"Market data shape: {market_data.shape}")
print(f"Duplicate dates in market data: {market_data.duplicated(subset=['dlycaldt']).sum()}")

# Show sample of selected data
print(f"\nSample of selected data:")
print(selected_data[['ticker', 'dlycaldt', 'dlyret']].head(10))

# %%
"""
## Calculating Cumulative Returns

Now let's calculate cumulative returns for our selected stocks to analyze their performance over time.
"""

# %%
def calculate_cumulative_returns(data, return_col='dlyret'):
    """Calculate cumulative returns for each stock"""
    data = data.copy()
    data = data.sort_values(['ticker', 'dlycaldt'])
    
    # Handle potential duplicate dates by taking the last observation for each ticker-date combination
    data = data.drop_duplicates(subset=['ticker', 'dlycaldt'], keep='last')
    
    # Calculate cumulative returns (1 + return) for each stock using a safer approach
    data['cumret'] = 1.0  # Initialize with 1
    
    for ticker in data['ticker'].unique():
        mask = data['ticker'] == ticker
        returns = data.loc[mask, return_col].fillna(0)
        cumulative = (1 + returns).cumprod()
        data.loc[mask, 'cumret'] = cumulative
    
    return data

# %%
# Calculate cumulative returns
stock_cumret = calculate_cumulative_returns(selected_data)
market_cumret = market_data.copy()
market_cumret['cumret'] = (1 + market_cumret['sprtrn']).cumprod()

# %%
"""
## Plot 1: Matplotlib - Cumulative Returns Comparison

Let's start with a traditional matplotlib plot to visualize the cumulative returns.
"""

# %%
# Plot 1: Matplotlib - Cumulative Returns Comparison
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8')

for ticker in selected_stocks:
    ticker_data = stock_cumret[stock_cumret['ticker'] == ticker]
    plt.plot(ticker_data['dlycaldt'], ticker_data['cumret'], 
             label=ticker, linewidth=2, marker='o', markersize=3)

# Add market portfolio
plt.plot(market_cumret['dlycaldt'], market_cumret['cumret'], 
         label='S&P 500', linewidth=3, color='black', linestyle='--')

plt.title('Cumulative Returns Comparison (2019-2025)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Return (1 = $1 invested)', fontsize=12)
plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
"""
## Plot 2: Seaborn - Cumulative Returns with Better Styling

Now let's use Seaborn for enhanced styling and aesthetics.
"""

# %%
# Plot 2: Seaborn - Cumulative Returns with better styling
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
sns.set_palette("husl")

# Prepare data for seaborn
plot_data = []
for ticker in selected_stocks:
    ticker_data = stock_cumret[stock_cumret['ticker'] == ticker][['dlycaldt', 'cumret']]
    ticker_data['ticker'] = ticker
    plot_data.append(ticker_data)

# Add market data
market_plot_data = market_cumret[['dlycaldt', 'cumret']].copy()
market_plot_data['ticker'] = 'S&P 500'
plot_data.append(market_plot_data)

plot_df = pd.concat(plot_data, ignore_index=True)

sns.lineplot(data=plot_df, x='dlycaldt', y='cumret', hue='ticker', 
             linewidth=2, markers=True, markersize=4)

plt.title('Cumulative Returns: Selected Stocks vs S&P 500', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Return', fontsize=12)
plt.legend(title='Asset', fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
"""
## Plot 3: Plotly Express - Interactive Cumulative Returns

Finally, let's create an interactive plot using Plotly Express for enhanced user experience.
"""

# %%
# Plot 3: Plotly Express - Interactive Cumulative Returns
fig = px.line(plot_df, x='dlycaldt', y='cumret', color='ticker',
              title='Interactive Cumulative Returns Comparison',
              labels={'dlycaldt': 'Date', 'cumret': 'Cumulative Return', 'ticker': 'Asset'},
              line_shape='linear', render_mode='svg')

fig.update_layout(
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12,
    legend_title_font_size=12,
    hovermode='x unified'
)

fig.show()

# %%
"""
## Dividend Analysis

Now let's analyze dividends to understand the income component of our selected stocks.
"""

# %%
# Now let's analyze dividends
print("\n=== Dividend Analysis ===")

# %%
def calculate_cumulative_dividends(data):
    """Calculate cumulative dividends for each stock"""
    data = data.copy()
    data = data.sort_values(['ticker', 'dlycaldt'])
    
    # Handle potential duplicate dates by taking the last observation for each ticker-date combination
    data = data.drop_duplicates(subset=['ticker', 'dlycaldt'], keep='last')
    
    # Sum up all dividend amounts (ordinary + non-ordinary)
    data['total_div'] = data['dlyorddivamt'].fillna(0) + data['dlynonorddivamt'].fillna(0)
    
    # Calculate cumulative dividends using a safer approach
    data['cumdiv'] = 0.0  # Initialize with 0
    
    for ticker in data['ticker'].unique():
        mask = data['ticker'] == ticker
        dividends = data.loc[mask, 'total_div']
        cumulative = dividends.cumsum()
        data.loc[mask, 'cumdiv'] = cumulative
    
    return data

# %%
# Calculate cumulative dividends
stock_cumdiv = calculate_cumulative_dividends(selected_data)

# Get dividend summary
div_summary = stock_cumdiv.groupby('ticker').agg({
    'total_div': ['sum', 'count'],
    'cumdiv': 'max'
}).round(4)

div_summary.columns = ['total_dividends', 'dividend_days', 'cumulative_dividends']
print(div_summary)

# %%
"""
## Plot 4: Matplotlib - Cumulative Dividends

Let's visualize the cumulative dividends using matplotlib.
"""

# %%
# Plot 4: Matplotlib - Cumulative Dividends
plt.figure(figsize=(12, 8))

for ticker in selected_stocks:
    ticker_data = stock_cumdiv[stock_cumdiv['ticker'] == ticker]
    plt.plot(ticker_data['dlycaldt'], ticker_data['cumdiv'], 
             label=f'{ticker} (Total: ${ticker_data["cumdiv"].max():.2f})', 
             linewidth=2, marker='s', markersize=3)

plt.title('Cumulative Dividends Paid (2019-2025)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Dividends ($)', fontsize=12)
plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
"""
## Plot 5: Seaborn - Dividend Comparison

Now let's use Seaborn for the dividend visualization.
"""

# %%
# Plot 5: Seaborn - Dividend Comparison
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Prepare dividend data for seaborn
div_plot_data = []
for ticker in selected_stocks:
    ticker_data = stock_cumdiv[stock_cumdiv['ticker'] == ticker][['dlycaldt', 'cumdiv']]
    ticker_data['ticker'] = ticker
    div_plot_data.append(ticker_data)

div_plot_df = pd.concat(div_plot_data, ignore_index=True)

sns.lineplot(data=div_plot_df, x='dlycaldt', y='cumdiv', hue='ticker', 
             linewidth=2, markers=True, markersize=4)

plt.title('Cumulative Dividends: Selected Stocks', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Dividends ($)', fontsize=12)
plt.legend(title='Stock', fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
"""
## Plot 6: Plotly Express - Interactive Dividends

Let's create an interactive dividend plot with Plotly.
"""

# %%
# Plot 6: Plotly Express - Interactive Dividends
fig = px.line(div_plot_df, x='dlycaldt', y='cumdiv', color='ticker',
              title='Interactive Cumulative Dividends',
              labels={'dlycaldt': 'Date', 'cumdiv': 'Cumulative Dividends ($)', 'ticker': 'Stock'},
              line_shape='linear', render_mode='svg')

fig.update_layout(
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12,
    legend_title_font_size=12,
    hovermode='x unified'
)

fig.show()

# %%
"""
## Rolling Volatility Analysis

Now let's analyze the rolling volatility of our selected stocks to understand their risk characteristics over time.
"""

# %%
# Now let's calculate rolling volatility (3-month window)
print("\n=== Rolling Volatility Analysis ===")

# %%
def calculate_rolling_volatility(data, window_days=63):  # ~3 months (63 trading days)
    """Calculate rolling volatility for each stock"""
    data = data.copy()
    data = data.sort_values(['ticker', 'dlycaldt'])
    
    # Handle potential duplicate dates by taking the last observation for each ticker-date combination
    data = data.drop_duplicates(subset=['ticker', 'dlycaldt'], keep='last')
    
    # Calculate rolling standard deviation of returns using a safer approach
    data['rolling_vol'] = np.nan
    data['rolling_vol_annual'] = np.nan
    
    for ticker in data['ticker'].unique():
        mask = data['ticker'] == ticker
        returns = data.loc[mask, 'dlyret'].fillna(0)
        
        # Calculate rolling standard deviation
        rolling_std = returns.rolling(window=window_days, min_periods=30).std()
        data.loc[mask, 'rolling_vol'] = rolling_std
        
        # Annualize volatility (multiply by sqrt(252) for daily data)
        data.loc[mask, 'rolling_vol_annual'] = rolling_std * np.sqrt(252)
    
    return data

# %%
# Calculate rolling volatility for stocks
stock_vol = calculate_rolling_volatility(selected_data)

# Calculate rolling volatility for market
market_vol = market_data.copy()
market_vol['rolling_vol'] = market_vol['sprtrn'].rolling(
    window=63, min_periods=30
).std()
market_vol['rolling_vol_annual'] = market_vol['rolling_vol'] * np.sqrt(252)

# %%
"""
## Plot 7: Matplotlib - Rolling Volatility

Let's visualize the rolling volatility using matplotlib.
"""

# %%
# Plot 7: Matplotlib - Rolling Volatility
plt.figure(figsize=(12, 8))

for ticker in selected_stocks:
    ticker_data = stock_vol[stock_vol['ticker'] == ticker]
    plt.plot(ticker_data['dlycaldt'], ticker_data['rolling_vol_annual'] * 100, 
             label=ticker, linewidth=2, alpha=0.8)

# Add market volatility
plt.plot(market_vol['dlycaldt'], market_vol['rolling_vol_annual'] * 100, 
         label='S&P 500', linewidth=3, color='black', linestyle='--')

plt.title('Rolling 3-Month Volatility (Annualized)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volatility (%)', fontsize=12)
plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
"""
## Plot 8: Seaborn - Volatility Comparison

Now let's use Seaborn for the volatility visualization.
"""

# %%
# Plot 8: Seaborn - Volatility Comparison
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Prepare volatility data for seaborn
vol_plot_data = []
for ticker in selected_stocks:
    ticker_data = stock_vol[stock_vol['ticker'] == ticker][['dlycaldt', 'rolling_vol_annual']]
    ticker_data['ticker'] = ticker
    vol_plot_data.append(ticker_data)

# Add market volatility
market_vol_data = market_vol[['dlycaldt', 'rolling_vol_annual']].copy()
market_vol_data['ticker'] = 'S&P 500'
vol_plot_data.append(market_vol_data)

vol_plot_df = pd.concat(vol_plot_data, ignore_index=True)
vol_plot_df['rolling_vol_annual'] = vol_plot_df['rolling_vol_annual'] * 100  # Convert to percentage

sns.lineplot(data=vol_plot_df, x='dlycaldt', y='rolling_vol_annual', hue='ticker', 
             linewidth=2, alpha=0.8)

plt.title('Rolling 3-Month Volatility: Stocks vs S&P 500', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volatility (%)', fontsize=12)
plt.legend(title='Asset', fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
"""
## Plot 9: Plotly Express - Interactive Volatility

Finally, let's create an interactive volatility plot with Plotly.
"""

# %%
# Plot 9: Plotly Express - Interactive Volatility
fig = px.line(vol_plot_df, x='dlycaldt', y='rolling_vol_annual', color='ticker',
              title='Interactive Rolling Volatility Analysis',
              labels={'dlycaldt': 'Date', 'rolling_vol_annual': 'Volatility (%)', 'ticker': 'Asset'},
              line_shape='linear', render_mode='svg')

fig.update_layout(
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12,
    legend_title_font_size=12,
    hovermode='x unified'
)

fig.show()

# %%
"""
## Summary Statistics

Let's compile a comprehensive summary of our analysis results.
"""

# %%
# Summary statistics
print("\n=== Summary Statistics ===")
print("Cumulative Returns (as of latest date):")
for ticker in selected_stocks:
    latest_ret = stock_cumret[stock_cumret['ticker'] == ticker]['cumret'].iloc[-1]
    print(f"{ticker}: {latest_ret:.2f}x")

latest_market_ret = market_cumret['cumret'].iloc[-1]
print(f"S&P 500: {latest_market_ret:.2f}x")

print("\nTotal Dividends Paid:")
for ticker in selected_stocks:
    total_div = stock_cumdiv[stock_cumdiv['ticker'] == ticker]['cumdiv'].iloc[-1]
    print(f"{ticker}: ${total_div:.2f}")

print("\nAverage Annualized Volatility (3-month rolling):")
for ticker in selected_stocks:
    avg_vol = stock_vol[stock_vol['ticker'] == ticker]['rolling_vol_annual'].mean() * 100
    print(f"{ticker}: {avg_vol:.1f}%")

avg_market_vol = market_vol['rolling_vol_annual'].mean() * 100
print(f"S&P 500: {avg_market_vol:.1f}%")

# %%
"""
## Analysis Complete

This script demonstrates:
1. Matplotlib plotting with pyplot interface
2. Seaborn plotting with enhanced styling
3. Plotly Express for interactive visualizations
4. Cumulative return analysis for selected stocks vs S&P 500
5. Dividend analysis comparing dividend-paying vs non-dividend stocks
6. Rolling volatility analysis using 3-month windows

The analysis provides insights into:
- Performance comparison between different types of stocks
- Income generation through dividends
- Risk characteristics over time
- Interactive visualization capabilities for data exploration
"""

print("\n=== Analysis Complete ===")
print("This script demonstrates:")
print("1. Matplotlib plotting with pyplot interface")
print("2. Seaborn plotting with enhanced styling")
print("3. Plotly Express for interactive visualizations")
print("4. Cumulative return analysis for selected stocks vs S&P 500")
print("5. Dividend analysis comparing dividend-paying vs non-dividend stocks")
print("6. Rolling volatility analysis using 3-month windows")
