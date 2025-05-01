import time
import numpy as np
from stable_baselines3 import PPO
from PriceFetcher.BitcoinPriceFetcher import BitcoinPriceFetcher
from SentimentAnalyzer.NewsSentimentAnalyzer import NewsSentimentAnalyzer


model_path = "Agent/Agent_Swarna.zip"
model = PPO.load(model_path)


price_fetcher = BitcoinPriceFetcher()
sentiment_analyzer = NewsSentimentAnalyzer()


starting_balance = 35000
balance = starting_balance
crypto_held = 0
net_worth = starting_balance



while True:

    live_candle = price_fetcher.fetch_price()
    sentiment_score, _ = sentiment_analyzer.analyze_sentiment()

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


        obs[0:5] /= 100_000
        obs[5:8] /= 100_000

        action, _ = model.predict(obs, deterministic=False)
        action = float(action[0])

        if action > 0 and balance > 0:
            invest_amount = balance * min(action, 1.0)
            crypto_held += invest_amount / close_price
            balance -= invest_amount
            action_name = f"BUY {round(action*100)}%"

        elif action < 0 and crypto_held > 0:
            sell_amount = crypto_held * min(abs(action), 1.0)
            balance += sell_amount * close_price
            crypto_held -= sell_amount
            action_name = f"SELL {round(abs(action)*100)}%"

        else:
            action_name = "HOLD"

        net_worth = balance + (crypto_held * close_price)
        profit_loss = net_worth - starting_balance

        print(f"ACTION: {action_name}")
        print(f"Net Worth: ${net_worth:.2f}")
        print(f"BTC Held: {crypto_held:.6f}")
        print(f"Available Cash: ${balance:.2f}")
        print(f"BTC Price: ${close_price:.2f}")
        print(f"Sentiment Score: {sentiment_score:.4f}")
        print(f"Profit/Loss: ${profit_loss:.2f}")

    else:
        print("❗ Failed to get live price. Skipping this round.")

    time.sleep(15)
