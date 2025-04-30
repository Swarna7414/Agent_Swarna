import time
import numpy as np
from stable_baselines3 import PPO
from PriceFetcher.BitcoinPriceFetcher import BitcoinPriceFetcher
from SentimentAnalyzer.NewsSentimentAnalyzer import NewsSentimentAnalyzer

# Load trained model
model = PPO.load("./ppo_sentiment_trading_model.zip")

# Initialize services
price_fetcher = BitcoinPriceFetcher()
sentiment_analyzer = NewsSentimentAnalyzer()

# Initial state
starting_balance = 35000
balance = starting_balance
crypto_held = 0.0
net_worth = balance

# Main loop
while True:
    print("\n🔁 Fetching live market data...")

    live_price = price_fetcher.fetch_price()
    sentiment_score, _ = sentiment_analyzer.analyze_sentiment()

    if not live_price:
        print("❗ Failed to fetch live price data.")
        time.sleep(5)
        continue

    obs = np.array([
        live_price['open'],
        live_price['high'],
        live_price['low'],
        live_price['close'],
        live_price['volume'],
        balance,
        crypto_held,
        net_worth,
        sentiment_score
    ], dtype=np.float32)

    # Predict next action
    action, _ = model.predict(obs, deterministic=False)

    close_price = live_price['close']
    if action == 1 and balance > 0:
        crypto_held += balance / close_price
        balance = 0
        action_name = "BUY ✅"
    elif action == 2 and crypto_held > 0:
        balance += crypto_held * close_price
        crypto_held = 0
        action_name = "SELL ✅"
    else:
        action_name = "HOLD ✅"

    net_worth = balance + (crypto_held * close_price)
    profit_loss = net_worth - starting_balance

    print(f"⚡ ACTION: {action_name}")
    print(f"💵 Net Worth: ${net_worth:.2f} | ₿ BTC Held: {crypto_held:.6f}")
    print(f"📈 BTC Price: ${close_price:.2f} | 🧠 Sentiment: {sentiment_score:.4f}")
    print(f"📊 Profit/Loss: ${profit_loss:.2f}")

    time.sleep(5)