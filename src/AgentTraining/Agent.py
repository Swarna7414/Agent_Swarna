from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os



raw_env = DummyVecEnv([lambda: BitcoinTradingEnv(training_df)])
env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)


log_dir = r"D:\College Projects\BitCoin Trading Agent\logs\PPO_1"
model_folder = r"D:\College Projects\BitCoin Trading Agent\Agent\Agent_Swarna.zip"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)


model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=0.0001,
    n_steps=4096,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    n_epochs=10,
    tensorboard_log=log_dir
)


total_timesteps = 333_333
model.learn(total_timesteps=total_timesteps)


model_path = os.path.join(model_folder, "Agent_Swarna")
model.save(model_path)

vecnorm_path = os.path.join(model_folder, "vecnormalize.pkl")
env.save(vecnorm_path)
