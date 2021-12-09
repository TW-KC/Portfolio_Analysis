import pandas as pd
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import scipy.stats

start='2017-01-01'
end='2021-01-01'

# Tickers
def get_tickers(n):

    tickers = []
    for i in range(0, n):
        ticker = input(f'Please input ticker {i+1}: ')
        tickers.append(ticker.upper())
    
    return tickers

# Portfolio Data
def get_data(tickers, start=start, end=end):

    # Portfolio Returns Data
    portfolio = pd.DataFrame()
    portfolio_returns = pd.DataFrame()
    for ticker in tickers:
        portfolio[ticker] = pdr.DataReader(ticker, 'yahoo', start=start, end=end)['Adj Close']
        portfolio_returns[f'{ticker} Returns'] = np.log((portfolio[ticker])).pct_change()
        
    return portfolio_returns

# Market Data
def get_market(equities='^GSPC', bonds='AGG', start=start, end=end):
    
    # Market Returns
    equities = (np.log(pdr.DataReader(equities, 'yahoo', start=start, end=end)['Adj Close'])).pct_change()
    bonds = (np.log(pdr.DataReader(bonds, 'yahoo', start=start, end=end)['Adj Close'])).pct_change()
    
    return equities, bonds

# Sharpe Ratio
def sharpe(tickers, portfolio_returns, equities):
    # Sharpe Ratio
    sharpe_ratio = pd.DataFrame()
    for ticker in tickers:
        sharpe_ratio[ticker] = (portfolio_returns[f'{ticker} Returns'] - equities) / portfolio_returns[f'{ticker} Returns'].std()
    return sharpe_ratio

# Drawdowns
def drawdown(tickers, portfolio_returns):
    # Drawdowns 
    drawdown_dataframe = pd.DataFrame()
    for ticker in tickers:
        wealth_index = (1 + portfolio_returns[f'{ticker} Returns']).cumprod()
        previous_peak = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peak) / previous_peak
        drawdown_dataframe[f'{ticker} Wealth'] = wealth_index
        drawdown_dataframe[f'{ticker} Peaks'] = previous_peak
        drawdown_dataframe[f'{ticker} Drawdowns'] = drawdowns
    return drawdown_dataframe

# Summary Statistics
def summary_statistics(portfolio_returns):
    
    def skew(portfolio_returns):
        # Skewness
        demeaned_returns = portfolio_returns - portfolio_returns.mean()
        sigma = portfolio_returns.std(ddof=0)
        exponential = (demeaned_returns**3).mean()
        return exponential/sigma**3

    def kurt(portfolio_returns, fisher=True):
        #Kurtosis
        demeaned_returns = portfolio_returns - portfolio_returns.mean()
        sigma = portfolio_returns.std(ddof=0)
        exponential = (demeaned_returns**4).mean()
        if fisher==True:
            return exponential/sigma**4 - 3
        else:
            return exponential/sigma**4
    
    def jarque_bera_test(portfolio_returns):
        # Jarque-Bera Normality Test
        return len(portfolio_returns) * ((skew(portfolio_returns)**2) + (kurt(portfolio_returns)**2 / 4)) / 6
    
    # Summary Statistics
    summary = pd.DataFrame({
        'Mean': portfolio_returns.mean(),
        'Median': portfolio_returns.median(),
        'Minimum': portfolio_returns.min(),
        'Maximum': portfolio_returns.max(),
        'Volatility': portfolio_returns.std(),
        'Observations': len(portfolio_returns),
        'Skewness': (skew(portfolio_returns)).round(5),
        'Excess Kurtosis': (kurt(portfolio_returns, fisher=True)).round(5),
        'Jarque-Bera': (jarque_bera_test(portfolio_returns)).round(5),
        'P-Value': (scipy.stats.distributions.chi2.pdf(jarque_bera_test(portfolio_returns), 2)).round(5)
    })
    pf_average = portfolio_returns.mean(axis=1)
    pf_av_df = {
        'Mean': pf_average.mean(),
        'Median': pf_average.median(),
        'Minimum': pf_average.min(),
        'Maximum': pf_average.max(),
        'Volatility': pf_average.std(),
        'Observations': len(pf_average),
        'Skewness': (skew(pf_average)),
        'Excess Kurtosis': (kurt(pf_average, fisher=True)),
        'Jarque-Bera': (jarque_bera_test(pf_average)),
        'P-Value': (scipy.stats.distributions.chi2.pdf(jarque_bera_test(pf_average), 2))
    }
    summary.loc['Portfolio Average Returns'] = pf_av_df
    return summary

# Portfolio Plot
def portfolio_plot(portfolio_returns, equities, bonds):
    pf_average = portfolio_returns.mean(axis=1)
    with plt.style.context('dark_background'):
        plt.figure(figsize=(16,10))
        plt.plot(market.cumsum(), label='Equities Market')
        plt.plot(bonds.cumsum(), color='red', label='Fixed Income Market')
        plt.plot(pf_average.cumsum(), color='yellow', label='Portfolio')
        plt.legend()
        plt.title('Equities Market, Fixed Income Market, and the Portfolio', fontsize=16)
        plt.axhline(y=0)
        plt.grid(False)
        plt.show();