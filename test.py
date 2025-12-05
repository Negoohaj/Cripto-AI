import yfinance as yf

print("Testando conexão com Yahoo Finance...")

btc = yf.download("BTC-USD", period="1y", progress=False)
print(btc.head())
