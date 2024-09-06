# export PYTHONPATH=$PYTHONPATH:.
# python3.12 src/agents/multiple_buy_agent_expanding_window.py
# Need to run in the root dir so the mlruns directory is located in the root

import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from src.observations.trend_observation_rms import TrendObservationRMS
from src.rewards.non_zero_buy_reward import NonZeroBuyReward
from src.utils.environment import get_env
from src.utils.mlflow import MLflowCallback, LOG_DIR
from src.utils.tick_data import get_data, plot_tick_data
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def start():
    experiment_name = "PPO Trading Agent - Expanding Window"
    mlflow.set_experiment(experiment_name)

    # ALL CONSTANTS
    train_timesteps = 2
    n_eval_episodes = 100 # For both train and test, on evaluate_policy
    tick_data, max_profit = get_data(1000)  # Generate 1000 data points
    print(f"Tick data shape: {tick_data.shape}")
    print(f"Max profit: {max_profit}")

    trend_offset = [1, 2, 5]
    std_multiplier = 1
    trend_observation = TrendObservationRMS(trend_offset, std_multiplier)
    lot = 0.01 * 100_000
    max_order = 1
    initial_balance = 10_000
    mlflow_callback = MLflowCallback(check_freq=1000, name="training")
    mlflow_callback_eval_train = MLflowCallback(check_freq=100, name="eval_train")
    mlflow_callback_eval_test = MLflowCallback(check_freq=100, name="eval_test")

    with mlflow.start_run():
        mlflow.log_param("max_profit", max_profit * lot * max_order)
        mlflow.log_param("std_multiplier", std_multiplier)
        mlflow.log_param("trend_offset", trend_offset)

        tscv = TimeSeriesSplit(n_splits=5)
        for fold, (train_index, test_index) in enumerate(tscv.split(tick_data), 1):
            with mlflow.start_run(nested=True, run_name=f"Fold {fold}"):
                train_data = tick_data.iloc[train_index]
                test_data = tick_data.iloc[test_index]

                env_kwargs = {
                    "initial_balance": initial_balance,
                    "tick_data": train_data,
                    "reward_func": NonZeroBuyReward,
                    "observation": trend_observation,
                    "max_orders": max_order,
                    "lot": lot,
                    "start_index": train_index[0],
                }

                env = DummyVecEnv([lambda: get_env("MultipleBuyEnv", "src.environments.multiple_buy.multiple_buy_environment:MultipleBuyEnvironment", LOG_DIR, env_kwargs)])

                batch_size = len(train_data) - max(trend_offset)
                model_params = {
                    "learning_rate": 1e-3,
                    "n_steps": batch_size,
                    "batch_size": batch_size,
                    "n_epochs": 10,
                    "gamma": 0.95,
                    "gae_lambda": 0.90,
                    "clip_range": 0.1,
                    "ent_coef": 0.05,
                    "max_grad_norm": 0.5,
                    "vf_coef": 0.5,
                }

                log_metrics_from_dict(model_params)

                # plot_tick_data(train_data)
                model = PPO("MlpPolicy", env, verbose=1, **model_params)
                
                mlflow.log_params(model_params)

                print(f"Train: index={train_index}")
                # print(f"Train: data={train_data}")
                model.learn(total_timesteps=train_timesteps, callback=mlflow_callback)

                # Evaluate on train data
                mean_reward_train, std_reward_train = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, callback=mlflow_callback_eval_train)
                mlflow.log_param("max_profit", max_profit * lot * max_order) # Duplicate so it's easier to compare in each fold

                # Evaluate on test data
                print(f"Test:  index={test_index}")
                # print(f"Test:  data={test_data}")
                test_env_kwargs = env_kwargs.copy()
                test_env_kwargs["start_index"] = test_index[0]
                test_env_kwargs["tick_data"] = test_data
                test_env = DummyVecEnv([lambda: get_env("MultipleBuyEnv", "src.environments.multiple_buy.multiple_buy_environment:MultipleBuyEnvironment", LOG_DIR, test_env_kwargs)])
                
                # Debugging code to check current_index and min_periods
                # base_env = test_env.envs[0]  # Get the underlying environment
                # print(f"Initial current_index: {base_env.unwrapped.current_index}")
                # print(f"start_index: {base_env.unwrapped.start_index}")
                # print(f"max_index: {base_env.unwrapped.max_index}")

                mean_reward_test, std_reward_test = evaluate_policy(model, test_env, n_eval_episodes=n_eval_episodes, callback=mlflow_callback_eval_test)

                # print(f"Fold {fold}:")
                # print(f"Train - Mean reward: {mean_reward_train:.2f} +/- {std_reward_train:.2f}")
                # print(f"Test - Mean reward: {mean_reward_test:.2f} +/- {std_reward_test:.2f}")

                # Run a few test episodes and log equity and balance
                obs = test_env.reset()
                equity = []
                balance = []
                for _ in range(len(test_data)):
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = test_env.step(action)
                    equity.append(infos[0]['equity'])
                    balance.append(infos[0]['balance'])

                    if dones.any():
                        break

                mlflow.log_metric("test/final_equity", equity[-1])
                mlflow.log_metric("test/final_balance", balance[-1])
                mlflow.log_param("train_size", len(train_data))
                mlflow.log_param("test_size", len(test_data))
            
            # mlflow.log_metric("mean_reward_train", mean_reward_train, fold)
            # mlflow.log_metric("mean_reward_test", mean_reward_test, fold)
            # mlflow.log_metric("std_reward_train", std_reward_train, fold)
            # mlflow.log_metric("std_reward_test", std_reward_test, fold)

def log_metrics_from_dict(metrics_dict):
    for key, value in metrics_dict.items():
        mlflow.log_param(key, value)

if __name__ == "__main__":
    start()