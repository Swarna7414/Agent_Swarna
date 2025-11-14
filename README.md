---
title: Agent Swarna - Bitcoin Trading Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Agent Swarna - Bitcoin Trading Agent 🤖

An intelligent Bitcoin trading agent powered by Reinforcement Learning (PPO) and sentiment analysis. This agent makes real-time trading decisions based on live Bitcoin price data and news sentiment.

## Features

- 🤖 **Reinforcement Learning Agent**: Uses Proximal Policy Optimization (PPO) for trading decisions
- 📊 **Live Price Data**: Fetches real-time Bitcoin prices from OKX exchange
- 📰 **Sentiment Analysis**: Analyzes Bitcoin news headlines using VADER sentiment analysis
- 🎯 **Smart Trading**: Makes buy/sell/hold decisions based on market conditions and sentiment

## API Endpoints

### `/agent`
Get a trading decision from the agent based on current market conditions.

**Response:**
```json
{
  "action": "BUY 75%",
  "totalcash": "$35000.00",
  "BTC": "0.123456",
  "moneyleft": "$8750.00",
  "BTCliveprice": "$45000.00",
  "Sentiment": "0.6543",
  "Profit_Loss": "$0.00"
}
```

### `/news`
Get latest Bitcoin news headlines.

**Response:**
```json
{
  "headlines": [
    "Bitcoin reaches new all-time high",
    "Major adoption news..."
  ]
}
```

### `/sentiment`
Get current sentiment score for Bitcoin news.

**Response:**
```json
{
  "sentiment": 0.654
}
```

### `/health`
Health check endpoint for monitoring and deployment pipelines.

**Response:**
```json
{
  "status": "healthy",
  "agent": "loaded",
  "price_fetcher": "ready",
  "sentiment_analyzer": "ready"
}
```

## How It Works

1. **Price Fetching**: The agent fetches live 1-minute candle data from OKX exchange
2. **Sentiment Analysis**: Analyzes recent Bitcoin news headlines to gauge market sentiment
3. **Decision Making**: The trained PPO model processes the observation vector (price data, balance, sentiment) and outputs a trading action
4. **Action Execution**: Based on the action, the agent buys, sells, or holds Bitcoin

## Technical Details

- **RL Algorithm**: PPO (Proximal Policy Optimization)
- **Observation Space**: 9 features (OHLCV, balance, crypto_held, net_worth, sentiment)
- **Action Space**: Continuous [-1, 1] where positive values indicate buy and negative values indicate sell
- **Initial Balance**: $35,000

## Deployment

This application is deployed on Hugging Face Spaces using Docker. The Dockerfile is configured to:
- Use Python 3.10
- Install all required dependencies
- Run the FastAPI application on port 7860

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
uvicorn Swarna:app --host=0.0.0.0 --port=7860
```

## Note

This is a demonstration project. Trading decisions are made by an AI agent and should not be used for actual trading without proper risk management and testing.

