import yfinance as yf

def load_data(market="GOOGL", start="2024-10-01", end="2025-06-06"):
    data = yf.download(market, start=start, end=end, interval="1d")

    # feature engineering 
    data['daily_return'] =  data['Close'].pct_change()
    data['volatility'] = (data['High'] - data['Low'])/(data['Open'])
    data['price_range'] = data['High'] - data['Low']
    data['close_open_diff'] = data['Close'] - data['Open']
    data['volume_change'] = data['Volume'].pct_change()

    # 1 = market price went up, 0 market price went down
    data['target'] =  (data['Close'].shift(-1) > data['Close']).astype(int)

    # remove last and first row cuz we dont have a time machine
    data = data[1:-1]
    X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return', 'volatility', 'price_range', 'close_open_diff', 'volume_change']].values
    Y = data['target'].values
    return X, Y
