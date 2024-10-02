# THIS AGENT IS NOT USING BASE_AGENT YET
# PROBABLY WILL BE DEPRECATED

# export PYTHONPATH=$PYTHONPATH:.
# python3.12 src/agents/multiple_buy_agent.py
# Need to run in the root dir so the mlruns directory is located in the root

import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
# from src.observations.trend_observation import TrendObservation
# from src.observations.trend_observation_percentage import TrendObservationPercentage
from src.observations.trend_observation_rms import TrendObservationStd
from src.rewards.non_zero_buy_reward import NonZeroBuyReward
from src.utils.environment import get_env
from src.callbacks.log_test_callback import LogTestCallback, LOG_DIR
from src.utils.tick_data import get_data, plot_tick_data
import pandas as pd


def start():
    experiment_name = "PPO Trading Agent - MultipleBuyAgent"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        data, max_profit = get_data()
        print(f"Tick data:\n{data}")
        print(f"Max profit: {max_profit}")

        trend_offset = [1, 2, 5]
        std_multiplier = 1
        trend_observation = TrendObservationStd(trend_offset, std_multiplier)
        lot = 0.01 * 100_000
        max_order = 5
        env_kwargs = {
            "initial_balance": 10_000,
            "data": data,
            "reward_func": NonZeroBuyReward,
            "observation": trend_observation,
            "max_orders": max_order,
            "lot": lot
        }
        mlflow.log_param("max_profit", max_profit * lot * max_order)
        mlflow.log_param("std_multiplier", std_multiplier)
        mlflow.log_param("trend_offset", trend_offset)

        env = DummyVecEnv(
            [
                lambda: get_env(
                    "MultipleBuyEnv",
                    "src.environments.multiple_buy.multiple_buy_environment:MultipleBuyEnvironment",
                    LOG_DIR,
                    env_kwargs,
                )
            ]
        )

        batch_size = len(data)-max(trend_offset)
        model_params = {
            "learning_rate": 1e-3,  # Increased for faster learning on small dataset
            "n_steps": batch_size,  # Set to the length of your data
            "batch_size": batch_size,  # Reduced due to small dataset
            "n_epochs": 10,  # Reduced to prevent overfitting
            "gamma": 0.95,  # Slightly reduced for shorter-term focus
            "gae_lambda": 0.90,  # Slightly reduced for shorter-term advantage estimation
            "clip_range": 0.1,  # Reduced to prevent too large policy updates on small data
            "ent_coef": 0.05,  # Increased to encourage exploration in limited data
            "max_grad_norm": 0.5,  # Added to prevent explosive gradients
            "vf_coef": 0.5,  # Added to balance value function and policy learning
        }

        log_metrics_from_dict(model_params)

        # plot_tick_data(data)
        # Create the PPO agent
        model = PPO("MlpPolicy", env, verbose=1, **model_params)

        # Set up MLflow callback
        mlflow_callback = LogTestCallback(check_freq=10000)

        # Log parameters
        mlflow.log_params(model_params)

        # Train the agent
        model.learn(total_timesteps=200_000, callback=mlflow_callback)

        # Evaluate the trained agent
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Log final metrics
        mlflow.log_metric("eval/mean_reward", mean_reward)
        mlflow.log_metric("eval/std_reward", std_reward)

        # Run a few test episodes
        obs = env.reset()
        for _ in range(100):  # Run for 100 steps
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            if dones.any():
                print(infos)
                obs = env.reset()
                break


def log_metrics_from_dict(metrics_dict):
    """
    Log all key-value pairs from a dictionary as metrics in MLflow.

    Args:
    metrics_dict (dict): A dictionary containing metric names as keys and metric values as values.
    """
    for key, value in metrics_dict.items():
        mlflow.log_param(key, value)

if __name__ == "__main__":
    start()
    # df = pd.read_csv('../data/tickdata_eurusd.csv')
    # print(df.head())
