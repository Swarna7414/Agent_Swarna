from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as panda
from src.ENV.BitcoinTradingEnv import BitcoinTradingEnv
import os


df = panda.read_csv(r"D:\College Projects\BitCoin Trading Agent\src\Data\TrainingData_with_indicators.csv", index_col='datetime', parse_dates=True)


env = DummyVecEnv([lambda: BitcoinTradingEnv(df)])


log_dir = r"D:\College Projects\BitCoin Trading Agent\logs\PPO_1"
os.makedirs(log_dir, exist_ok=True)


agent = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
    gamma=0.99,
    tensorboard_log=log_dir
)


agent.learn(total_timesteps=13000)


agentpath = r"D:\College Projects\BitCoin Trading Agent\Agent\Agent_Swarna.zip"
agent.save(agentpath)
