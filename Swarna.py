from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from stable_baselines3 import PPO
from PriceFetcher.BitcoinPriceFetcher import BitcoinPriceFetcher
from SentimentAnalyzer.NewsSentimentAnalyzer import NewsSentimentAnalyzer

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


agent = PPO.load("Agent/Agent_Swarna.zip")
price_fetcher = BitcoinPriceFetcher()
sentiment_analyzer = NewsSentimentAnalyzer()


starting_balance = 35000
balance = starting_balance
crypto_held = 0
net_worth = starting_balance

@app.get("/agent")
def trade():
    global balance, crypto_held, net_worth

    live_candle = price_fetcher.fetch_price()
    sentiment_score, _ = sentiment_analyzer.analyze_sentiment()

    if not live_candle:
        return {"error": "Failed to fetch live BTC data."}

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


    action, _ = agent.predict(obs, deterministic=False)
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

    return {
        "action": action_name,
        "totalcash": f"${net_worth:.2f}",
        "BTC": f"{crypto_held:.6f}",
        "moneyleft": f"${balance:.2f}",
        "BTCliveprice": f"${close_price:.2f}",
        "Sentiment": f"{sentiment_score:.4f}",
        "Profit_Loss": f"${profit_loss:.2f}"
    }

@app.get("/news")
def get_news_headlines():
    try:
        headlines = sentiment_analyzer.fetch_headlines()
        return {"headlines": headlines}
    except Exception as e:
        return {"error": str(e)}

@app.get("/sentiment")
def get_sentiment_score():
    try:
        sentiment_score, _ = sentiment_analyzer.analyze_sentiment()
        return {"sentiment": round(sentiment_score, 3)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    try:
        
        agent_check = agent is not None
        
        price_check = price_fetcher is not None
        
        sentiment_check = sentiment_analyzer is not None
        
        if agent_check and price_check and sentiment_check:
            return {
                "status": "healthy",
                "agent": "loaded",
                "price_fetcher": "ready",
                "sentiment_analyzer": "ready"
            }
        else:
            return {
                "status": "unhealthy",
                "agent": "loaded" if agent_check else "not loaded",
                "price_fetcher": "ready" if price_check else "not ready",
                "sentiment_analyzer": "ready" if sentiment_check else "not ready"
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}
