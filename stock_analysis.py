import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# download stock data
stock = "AAPL"

data = yf.download(stock, start="2020-01-01", end="2026-01-01")

print(data.head())

print(data.info())

print(data.describe())

plt.figure(figsize=(12,6))
plt.plot(data['Close'])
plt.title("Stock Closing Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

data['MA50'] = data['Close'].rolling(window=50).mean()

data['MA200'] = data['Close'].rolling(window=200).mean()

plt.figure(figsize=(12,6))

plt.plot(data['Close'], label="Close Price")
plt.plot(data['MA50'], label="50-Day MA")
plt.plot(data['MA200'], label="200-Day MA")

plt.legend()
plt.show()

data['Daily Return'] = data['Close'].pct_change()

print(data['Daily Return'])

plt.figure(figsize=(12,6))
plt.plot(data['Daily Return'])
plt.title("Daily Returns")
plt.show()

sns.histplot(data['Daily Return'].dropna(), bins=50)
plt.title("Return Distribution")
plt.show()

stocks = ['AAPL', 'TSLA', 'MSFT']

portfolio = yf.download(stocks, start="2020-01-01")['Close']

print(portfolio.head())

portfolio.plot(figsize=(12,6))
plt.title("Multiple Stock Comparison")
plt.show()

volatility = data['Daily Return'].std()

print("Volatility:", volatility)

data.to_csv("stock_data.csv")

print("Highest Price:", data['Close'].max())
print("Lowest Price:", data['Close'].min())
print("Average Price:", data['Close'].mean())
