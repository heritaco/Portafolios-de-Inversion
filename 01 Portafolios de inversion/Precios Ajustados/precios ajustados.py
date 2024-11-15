# pip install yfinance
import yfinance as yf
# pip install pandas
import pandas as pd

df = pd.DataFrame()     # Creamos una tabla vacía

# Lista de tickers
tickers = ['NKE', 'DIS', 'MSFT', 'COST', 'AMZN', 'AAPL', 'TSLA', 'NFLX', 'GOOGL', 'SBUX', 'KO', 'AAPL', 'COST']

for ticker in tickers:
    stock = yf.download(ticker, start='2014-10-18', end='2024-10-18') # Descargamps los datos de Yahoo Finance
    stock_adj_close = stock['Adj Close']            # Seleccionamos la columna Adj Close
    stock_adj_close.name = ticker                   # Renombramos la columna con el ticker
    df = pd.concat([df, stock_adj_close], axis=1)   # Concatenamos la columna a la tabla

df.to_excel('precios_ajustados.xlsx')               # Guardamos el la tabla en Excel