import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to fetch user input for tickers
input_str = input("Enter tickers of the stocks (separated by space): ")
elements = input_str.split()
tickers = [str(element) for element in elements]

print("Array entered by the user:", tickers)

# Setting end date and start date
end_date = datetime.today()
start_date = end_date - timedelta(days=5 * 365)

# Fetching adjusted close prices for each stock
adj_close_df = pd.DataFrame()

for stock in tickers:
    data = yf.download(stock, start=start_date, end=end_date)
    adj_close_df[stock] = data['Adj Close']

# Calculating log returns
log_return = np.log(adj_close_df / adj_close_df.shift(1))
log_return = log_return.dropna()  # Dropping missing values

# Finding Covariance matrix
cov_matrix = log_return.cov() * 252

# Function to calculate portfolio volatility
def portfolio_volatility(weights, cov_matrix):
    portfolio_variance = weights.T @ cov_matrix @ weights
    return np.sqrt(portfolio_variance)

def min_variance_objective(weights, cov_matrix):
    return portfolio_volatility(weights, cov_matrix)**2

cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = [(0, 0.5) for _ in range(len(tickers))]

# Initial weights
initial_weights = np.array([1 / len(tickers)] * len(tickers))


# Optimization for minimum variance
min_variance_result = minimize(min_variance_objective, initial_weights, args=(cov_matrix,),
                               method='SLSQP', constraints=cons, bounds=bounds)

optimal_weights_mv = min_variance_result.x

print('\nOptimal Weights (Minimum Variance):')
for ticker, weight in zip(tickers, optimal_weights_mv):
    print(f"{ticker}: {weight:.4f}")

# Calculating portfolio return and volatility
portfolio_return_mv = np.sum(log_return.mean() * optimal_weights_mv) * 252
portfolio_volatility_mv = portfolio_volatility(optimal_weights_mv, cov_matrix)

print(f"Portfolio Return (Minimum Variance): {portfolio_return_mv:.4f}")
print(f"Portfolio Volatility (Minimum Variance): {portfolio_volatility_mv:.4f}")

#plotting graph
plt.figure(figsize=(10,6))
plt.bar(tickers,optimal_weights_mv)
plt.xlabel('assets')
plt.ylabel('Optimal weights')
plt.title('Optimal Portfolio')
print(plt.show())