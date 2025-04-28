import requests

class BitcoinPriceFetcher:
    def __init__(self, inst_id='BTC-USDT', bar='1m'):
        self.url = 'https://www.okx.com/api/v5/market/candles'
        self.inst_id = inst_id
        self.bar = bar

    def fetch_price(self):
        params = {'instId': self.inst_id, 'bar': self.bar, 'limit': 1}
        try:
            response = requests.get(self.url, params=params)
            response.raise_for_status()
            data = response.json()
            if data['code'] != '0':
                print(f"Error from OKX API: {data['msg']}")
                return None
            latest_candle = data['data'][0]
            return {
                'open': float(latest_candle[1]),
                'high': float(latest_candle[2]),
                'low': float(latest_candle[3]),
                'close': float(latest_candle[4]),
                'volume': float(latest_candle[5])
            }
        except Exception as ex:
            print(f"Error fetching OKX data: {ex}")
            return None

# price = BitcoinPriceFetcher()
# live = price.fetch_price()
#
# if live:
#     print("open : ",live['open'])
#     print("close:",live['high'])
#     print("low:",live['low'])
#     print("close",live['close'])
#     print("voume:",live['volume'])
# else:
#     print("No live data fetched!")