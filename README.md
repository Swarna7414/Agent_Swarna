---
title: Agent Swarna
colorFrom: blue
colorTo: purple
sdk: docker
app_file: Swarna.py
pinned: false
---

# Agent Swarna

This project employs Reinforcement Learning (RL) to generate Bitcoin trading suggestions (Buy/Hold/Sell). By training an RL agent on historical data and integrating real-time Bitcoin prices along with relevant news, the system aims to provide users with intelligent and timely guidance on potential trading actions.

## Features

- **Reinforcement Learning Agent**: Trained using PPO (Proximal Policy Optimization) algorithm
- **Real-time Bitcoin Price Fetching**: Gets current Bitcoin prices from external APIs
- **News Sentiment Analysis**: Analyzes news sentiment to inform trading decisions
- **RESTful API**: FastAPI-based endpoints for interacting with the agent

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /agent` - Get trading suggestion from the RL agent
- Additional endpoints available in the FastAPI application

## How It Works

The agent combines:
1. Historical price data analysis
2. Real-time Bitcoin price information
3. News sentiment analysis
4. Reinforcement learning model predictions

To generate intelligent trading suggestions (Buy/Hold/Sell) based on market conditions.
