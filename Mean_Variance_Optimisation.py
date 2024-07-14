import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from fredapi import Fred
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#tickers of stock we want to optimize where SPY is  BND is GLD is gold QQQ is VTI is
#tickers = ['SPY','BND','GLD','QQQ','VTI']
# Function to take tickers input from user
input_str = input("Enter tickers of the stock: ")
elements = input_str.split()
tickers = [str(element) for element in elements]

print("Array entered by the user:", tickers)

#setting end date and start date
end_date = datetime.today()
start_date = end_date - timedelta(days= 5*365)
print(start_date)

#getting adjusted close price for every stock
adj_close_df = pd.DataFrame()

for stock in tickers:
    data = yf.download(stock, start= start_date, end= end_date)
    adj_close_df[stock] = data['Adj Close']
#check
""""
for stock in tickers:
    print( adj_close_df[stock])
"""
#calculating logNormal returns for each stock which is basically difference between natural log return of stock prices
# at two different time
log_return = np.log(adj_close_df/adj_close_df.shift(1))
log_return = log_return.dropna() #Dropping missing values

#Finding Covariance between two stocks
cov_matrix = log_return.cov()*252
#check
#print(cov_matrix)

#calculating portfolio variance and standard deviation
def standard_deviation (weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)      #standard deviation is square root of variance

#calculating expected returns assuming average of past returns
def expected_return (weights,log_return):
    return np.sum(log_return.mean()*weights)*252

#function to calculate sharp ratio
def sharp_ratio(weights, log_return, cov_matrix, risk_free_rate):
    return((expected_return (weights,log_return)-risk_free_rate)/standard_deviation (weights, cov_matrix))

#1b3896dbdbd6878090ae00d80913a34b
#Getting risk_free_rate using Fred_Api
fred = Fred(api_key = '1b3896dbdbd6878090ae00d80913a34b')
ten_year_treasury_rate = fred.get_series_latest_release('GS10')/100
risk_free_rate = ten_year_treasury_rate.iloc[-1]
#check
#print(risk_free_rate)


def neg_sharp_ratio(weights, log_return, cov_matrix, risk_free_rate):
    return -sharp_ratio(weights, log_return, cov_matrix, risk_free_rate)

cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights)-1}) #condition
bounds = [(0,0.5) for _ in range(len(tickers))] #boundry

#setting initial weights
initial_weights = np.array([1/len(tickers)]*len(tickers))
#check
#print(initial_weights)

optimised_result = minimize(neg_sharp_ratio,initial_weights,args=(log_return,cov_matrix,risk_free_rate),method='SLSQP',constraints=cons,bounds=bounds)

optimal_weight = optimised_result.x

print('Optimal Weights:')
for ticker,weight in zip(tickers,optimal_weight):
    print(f"{ticker}:{weight:.4f}")

optimal_portfolio_return = expected_return(optimal_weight, log_return)
optimal_portfolio_volatility = standard_deviation(optimal_weight, cov_matrix)
optimal_sharp_ratio = sharp_ratio(optimal_weight, log_return, cov_matrix, risk_free_rate)

print(f"Annual Return: {optimal_portfolio_return:.4f}")
print(f"Volatility: {optimal_portfolio_volatility:.4f}")
print(f"Sharp Ratio: {optimal_sharp_ratio:.4f}")

#plotting graph
plt.figure(figsize=(10,6))
plt.bar(tickers,optimal_weight)
plt.xlabel('assets')
plt.ylabel('Optimal weights')
plt.title('Optimal Portfolio')
print(plt.show())