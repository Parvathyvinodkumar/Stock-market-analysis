# ============================================
# ADVANCED STOCK MARKET ANALYSIS USING PYTHON
# ============================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')

# STEP 1: SELECT STOCKS

stocks = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN']

start_date = "2020-01-01"
end_date = "2026-01-01"

# Download data
data = yf.download(stocks, start=start_date, end=end_date)

close = data['Close']

print("\nStock Data Loaded Successfully\n")
print(close.head())

# STEP 2: BASIC STATISTICS

print("\nBasic Statistics:\n")
print(close.describe())

# STEP 3: MOVING AVERAGES (Example: AAPL)

aapl = yf.download("AAPL", start=start_date, end=end_date)

aapl['MA50'] = aapl['Close'].rolling(window=50).mean()
aapl['MA200'] = aapl['Close'].rolling(window=200).mean()

plt.figure(figsize=(12,6))

plt.plot(aapl['Close'], label='Close Price')
plt.plot(aapl['MA50'], label='50-Day MA')
plt.plot(aapl['MA200'], label='200-Day MA')

plt.title("AAPL Moving Averages")
plt.legend()
plt.show()

# STEP 4: DAILY RETURNS

returns = close.pct_change()

plt.figure(figsize=(12,6))
returns.plot()
plt.title("Daily Returns of Stocks")
plt.show()

# STEP 5: VOLATILITY ANALYSIS

volatility = returns.std()

print("\nVolatility of Stocks:\n")
print(volatility)

volatility.plot(kind='bar', figsize=(10,5))
plt.title("Stock Volatility Comparison")
plt.show()

# STEP 6: RSI INDICATOR

def compute_RSI(data, window=14):
    
    delta = data['Close'].diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    
    RS = gain / loss
    
    RSI = 100 - (100 / (1 + RS))
    
    return RSI

aapl['RSI'] = compute_RSI(aapl)

plt.figure(figsize=(12,6))
plt.plot(aapl['RSI'])
plt.title("RSI Indicator (AAPL)")
plt.axhline(70)
plt.axhline(30)
plt.show()

# STEP 7: BOLLINGER BANDS

aapl['MA20'] = aapl['Close'].rolling(20).mean()
aapl['STD'] = aapl['Close'].rolling(20).std()

aapl['Upper'] = aapl['MA20'] + (2 * aapl['STD'])
aapl['Lower'] = aapl['MA20'] - (2 * aapl['STD'])

plt.figure(figsize=(12,6))

plt.plot(aapl['Close'], label='Close')
plt.plot(aapl['Upper'], label='Upper Band')
plt.plot(aapl['Lower'], label='Lower Band')
plt.plot(aapl['MA20'], label='MA20')

plt.title("Bollinger Bands (AAPL)")
plt.legend()
plt.show()

# STEP 8: PORTFOLIO ANALYSIS

portfolio_returns = returns.mean()

portfolio_volatility = returns.std()

print("\nPortfolio Mean Returns:\n")
print(portfolio_returns)

print("\nPortfolio Risk (Volatility):\n")
print(portfolio_volatility)

# STEP 9: RISK VS RETURN SCATTER PLOT

plt.figure(figsize=(10,6))

plt.scatter(portfolio_volatility, portfolio_returns)

for i in stocks:
    
    plt.text(portfolio_volatility[i], portfolio_returns[i], i)

plt.title("Risk vs Return")
plt.xlabel("Risk (Volatility)")
plt.ylabel("Return")

plt.show()

# STEP 10: CORRELATION HEATMAP

plt.figure(figsize=(10,8))

sns.heatmap(returns.corr(), annot=True, cmap='coolwarm')

plt.title("Stock Correlation Heatmap")

plt.show()

# STEP 11: CUMULATIVE RETURNS

cumulative_returns = (1 + returns).cumprod()

cumulative_returns.plot(figsize=(12,6))

plt.title("Cumulative Returns of Stocks")

plt.show()

# STEP 12: DISTRIBUTION PLOT

plt.figure(figsize=(10,6))

sns.histplot(returns['AAPL'].dropna(), bins=50)

plt.title("Return Distribution of AAPL")

plt.show()

# STEP 13: SAVE DATA

close.to_csv("stock_prices.csv")
returns.to_csv("stock_returns.csv")

print("\nData saved to CSV successfully.")

# STEP 14: FINAL INSIGHTS

print("\nFinal Insights:\n")

print("Highest AAPL Price:", aapl['Close'].max())
print("Lowest AAPL Price:", aapl['Close'].min())

print("\nProject Completed Successfully.")
