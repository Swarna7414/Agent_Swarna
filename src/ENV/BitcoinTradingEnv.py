import gym
from gym import spaces
import numpy as np
import random

class BitcoinTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(BitcoinTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = 35000
        self.transaction_fee_percent = 0.001
        self.max_steps = len(self.df) - 1

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.net_worth
        return self._next_observation()

    def _get_sentiment_score(self):
        return random.uniform(0, 1)

    def _next_observation(self):
        row = self.df.iloc[self.current_step]

        obs = np.array([
            row['Open'] / 100_000,
            row['High'] / 100_000,
            row['Low'] / 100_000,
            row['Close'] / 100_000,
            row['Volume'] / 1_000_000,
            self.balance / 100_000,
            self.crypto_held,
            self.net_worth / 100_000,
            self._get_sentiment_score()
        ])
        return obs

    def step(self, action):
        action = float(action[0])
        current_price = self.df.iloc[self.current_step]['Close']

        
        if action > 0 and self.balance > 0:
            invest_amount = self.balance * min(action, 1.0)
            fee = invest_amount * self.transaction_fee_percent
            self.crypto_held += (invest_amount - fee) / current_price
            self.balance -= invest_amount

        
        elif action < 0 and self.crypto_held > 0:
            sell_amount = self.crypto_held * min(abs(action), 1.0)
            proceeds = sell_amount * current_price
            fee = proceeds * self.transaction_fee_percent
            self.balance += proceeds - fee
            self.crypto_held -= sell_amount

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        reward = self.net_worth - self.prev_net_worth
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step} | Net Worth: {self.net_worth:.2f} | BTC Held: {self.crypto_held:.5f}")
