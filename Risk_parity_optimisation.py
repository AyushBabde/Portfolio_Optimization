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

# Function for risk parity objective
def risk_parity_objective(weights, cov_matrix):
    portfolio_vol = portfolio_volatility(weights, cov_matrix)
    risk_contribution = cov_matrix @ weights
    return np.sum(np.square(risk_contribution / portfolio_vol - 1))

# Constraints and bounds for optimization
cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = [(0, 0.5) for _ in range(len(tickers))]

# Initial weights
initial_weights = np.array([1 / len(tickers)] * len(tickers))

# Optimization
risk_parity_result = minimize(risk_parity_objective, initial_weights, args=(cov_matrix,),
                              method='SLSQP', constraints=cons, bounds=bounds)

optimal_weights_rp = risk_parity_result.x

print('Optimal Weights (Risk Parity):')
for ticker, weight in zip(tickers, optimal_weights_rp):
    print(f"{ticker}: {weight:.4f}")

# Calculating portfolio return and volatility
portfolio_return_rp = np.sum(log_return.mean() * optimal_weights_rp) * 252
portfolio_volatility_rp = portfolio_volatility(optimal_weights_rp, cov_matrix)

print(f"Portfolio Return (Risk Parity): {portfolio_return_rp:.4f}")
print(f"Portfolio Volatility (Risk Parity): {portfolio_volatility_rp:.4f}")

#plotting graph
plt.figure(figsize=(10,6))
plt.bar(tickers,optimal_weights_rp)
plt.xlabel('assets')
plt.ylabel('Optimal weights')
plt.title('Optimal Portfolio')
print(plt.show())