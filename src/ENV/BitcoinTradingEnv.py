import gym
from gym import spaces
import numpy as np
import pandas as pd

class BitcoinTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(BitcoinTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = 10000
        self.current_step = 0
        self.max_steps = len(self.df) - 1

        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        self.last_action_type = "HOLD"

        # 🎯 Continuous action: -1 to 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # 📈 Observation: OHLVC + account info (8 features)
        self.observation_space = spaces.Box(
            low=-1.0, high=np.inf, shape=(8,), dtype=np.float32
        )

    def _next_observation(self):
        row = self.df.iloc[self.current_step]

        return np.array([
            row['Open'],
            row['High'],
            row['Low'],
            row['Close'],
            row['Volume'],
            self.balance,
            self.crypto_held,
            self.net_worth
        ], dtype=np.float32)

    def step(self, action):
        action = float(action[0])
        current_price = self.df.iloc[self.current_step]['Close']
        threshold = 0.05

        # BUY
        if action > threshold and self.balance > 0:
            invest_amount = self.balance * min(action, 1.0)
            self.crypto_held += invest_amount / current_price
            self.balance -= invest_amount
            self.last_action_type = f"BUY {int(min(action * 100, 100))}%"

        # SELL
        elif action < -threshold and self.crypto_held > 0:
            sell_amount = self.crypto_held * min(abs(action), 1.0)
            self.balance += sell_amount * current_price
            self.crypto_held -= sell_amount
            self.last_action_type = f"SELL {int(min(abs(action) * 100, 100))}%"

        # HOLD
        else:
            self.last_action_type = "HOLD"

        self.net_worth = self.balance + self.crypto_held * current_price
        reward = self.net_worth - self.initial_balance

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self._next_observation(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        self.last_action_type = "HOLD"
        return self._next_observation(), {}

    def render(self, mode='human'):
        current_price = self.df.iloc[self.current_step]['Close']
        print(f"Step: {self.current_step}")
        print(f"Action: {self.last_action_type}")
        print(f"Price: ${current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"BTC Held: {self.crypto_held:.5f}")
        print(f"Net Worth: ${self.net_worth:.2f}")
