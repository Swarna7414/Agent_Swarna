import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO


df = pd.read_csv(r"D:\College Projects\BitCoin Trading Agent\src\Data\TestData.csv", parse_dates=['datetime'])
df.set_index('datetime', inplace=True)
df = df.reset_index()


df['date_only'] = df['datetime'].dt.date
unique_dates = sorted(df['date_only'].unique())
selected_dates = unique_dates[10:13]  # Days 11 to 13
phase_3_data = df[df['date_only'].isin(selected_dates)].copy().reset_index(drop=True)
timestamps = phase_3_data['datetime']


model_path = r"D:\College Projects\BitCoin Trading Agent\Agent\Agent_Swarna.zip"
model = PPO.load(model_path)


balance = 1000
crypto_held = 0
net_worth = 1000
net_worth_history = []
btc_held_history = []

for idx, row in phase_3_data.iterrows():
    if idx < len(phase_3_data) // 3:
        sentiment_score = 0.6
    elif idx < 2 * len(phase_3_data) // 3:
        sentiment_score = -0.3
    else:
        sentiment_score = 0.1

    obs = np.array([
        row['Open'],
        row['High'],
        row['Low'],
        row['Close'],
        row['Volume'],
        balance,
        crypto_held,
        net_worth,
        sentiment_score
    ], dtype=np.float32)

    action, _ = model.predict(obs, deterministic=False)

    if action == 1 and balance > 0:
        crypto_held += balance / row['Close']
        balance = 0
    elif action == 2 and crypto_held > 0:
        balance += crypto_held * row['Close']
        crypto_held = 0

    net_worth = balance + crypto_held * row['Close']
    net_worth_history.append(net_worth)
    btc_held_history.append(crypto_held)


print(f"✅ Final Net Worth after Phase 3 (Sentiment-Based): ${net_worth:.2f}")


plt.figure(figsize=(18, 6))
plt.plot(timestamps, net_worth_history, color='red', linewidth=2)
plt.title('Sentiment-Aware RL Agent - Net Worth Over Time (Phase 3)')
plt.xlabel('Time')
plt.ylabel('Net Worth ($)')
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 6))
plt.step(timestamps, btc_held_history, where='post', color='black', linewidth=2)
plt.title('BTC Held Over Time by Sentiment-Aware RL Agent (Phase 3)')
plt.xlabel('Time')
plt.ylabel('BTC Held')
plt.grid(True)
plt.tight_layout()
plt.show()
