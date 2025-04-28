import uvicorn
from fastapi import FastAPI
from PriceFetcher.BitcoinPriceFetcher import BitcoinPriceFetcher
from SentimentAnalyzer.NewsSentimentAnalyzer import NewsSentimentAnalyzer
from stable_baselines3 import PPO
import numpy as np

app = FastAPI()


price_fetcher = BitcoinPriceFetcher()
news_analyzer = NewsSentimentAnalyzer()
model = PPO.load("./ppo_sentiment_trading_model.zip")


starting_balance = 35000
balance = starting_balance
crypto_held = 0
net_worth = balance

@app.get("/status")
def get_status():
    global balance, crypto_held, net_worth

    live_candle = price_fetcher.fetch_price()
    sentiment_score, _ = news_analyzer.analyze_sentiment()

    if live_candle:
        open_price = live_candle['open']
        high_price = live_candle['high']
        low_price = live_candle['low']
        close_price = live_candle['close']
        volume = live_candle['volume']

        obs = np.array([
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
            balance,
            crypto_held,
            net_worth,
            sentiment_score
        ], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=False)


        if action == 1 and balance > 0:
            crypto_held += balance / close_price
            balance = 0
            action_name = "BUY"
        elif action == 2 and crypto_held > 0:
            balance += crypto_held * close_price
            crypto_held = 0
            action_name = "SELL"
        else:
            action_name = "HOLD"


        net_worth = balance + (crypto_held * close_price)
        profit_loss = net_worth - starting_balance


        return {
            "action": action_name,
            "net_worth": round(net_worth, 2),
            "btc_held": round(crypto_held, 6),
            "sentiment": round(sentiment_score, 4),
            "price": round(close_price, 2),
            "profit_loss": round(profit_loss, 2)
        }

    else:
        return {"error": "Due to API Error Couldn't able to get the Response"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)