import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


df = pd.read_csv(r"D:\College Projects\BitCoin Trading Agent\src\Data\TestData.csv", parse_dates=['datetime'])
df.set_index('datetime', inplace=True)
df = df.reset_index()


df['date_only'] = df['datetime'].dt.date
unique_dates = sorted(df['date_only'].unique())
selected_dates = unique_dates[5:10]
filtered_df = df[df['date_only'].isin(selected_dates)].drop(columns=['date_only'])


agent_path = r"D:\College Projects\BitCoin Trading Agent\Agent\Agent_Swarna.zip"
agent = PPO.load(agent_path)


filtered_df['date'] = filtered_df['datetime'].dt.date
unique_days = sorted(filtered_df['date'].unique())
colors = ['blue', 'green', 'red', 'orange', 'purple']

plt.figure(figsize=(18, 10))

for i, day in enumerate(unique_days):
    day_data = filtered_df[filtered_df['date'] == day].copy()


    day_data = day_data.iloc[::5].reset_index(drop=True)

    balance = 1000
    crypto_held = 0
    net_worth = 1000

    net_worth_history = []
    minute_counter = []

    for idx, row in enumerate(day_data.itertuples()):
        obs = np.array([
            row.Open,
            row.High,
            row.Low,
            row.Close,
            row.Volume,
            balance,
            crypto_held,
            net_worth,
            0
        ]).astype(np.float32)

        action, _ = agent.predict(obs, deterministic=False)

        if action == 1 and balance > 0:
            crypto_held += balance / row.Close
            balance = 0
        elif action == 2 and crypto_held > 0:
            balance += crypto_held * row.Close
            crypto_held = 0

        net_worth = balance + crypto_held * row.Close
        net_worth_history.append(net_worth)
        minute_counter.append(idx * 5)

    plt.plot(minute_counter, net_worth_history, label=f'Day {i+6}: {day}', color=colors[i])

plt.xlabel('Minutes into the Day (5-min steps)')
plt.ylabel('Net Worth ($)')
plt.title('Bitcoin RL Agent - Evaluation (Days 6–10, 5-Min Intervals)')
plt.legend()
plt.grid()
plt.show()
