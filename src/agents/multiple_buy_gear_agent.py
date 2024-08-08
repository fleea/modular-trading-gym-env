# export PYTHONPATH=$PYTHONPATH:.
# python3.12 src/agents/multiple_buy_gear_agent.py
# Need to run in the root dir so the mlruns directory is located in the root

import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from src.observations.trend_observation import TrendObservation
from src.rewards.non_zero_buy_reward import NonZeroBuyReward
from src.utils.environment import get_env
from src.utils.mlflow import MLflowCallback, LOG_DIR
from src.utils.tick_data import get_data


def start():
    experiment_name = "PPO Trading Agent - MultipleBuyGearAgent"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        tick_data = get_data()
        print(f"Tick data:\n{tick_data}")

        trend_observation = TrendObservation([1])
        env_kwargs = {
            "initial_balance": 10_000,
            "tick_data": tick_data,
            "reward_func": NonZeroBuyReward,
            "observation": trend_observation,
            "max_orders": 1,
        }

        env = DummyVecEnv(
            [
                lambda: get_env(
                    "MultipleBuyGearEnv",
                    "src.environments.multiple_buy_gear.multiple_buy_gear_environment:MultipleBuyGearEnvironment",
                    LOG_DIR,
                    env_kwargs,
                )
            ]
        )

        model_params = {
            "learning_rate": 1e-3,  # Increased for faster learning on small dataset
            "n_steps": 300,  # Set to the length of your data
            "batch_size": 60,  # Reduced due to small dataset
            "n_epochs": 10,  # Reduced to prevent overfitting
            "gamma": 0.95,  # Slightly reduced for shorter-term focus
            "gae_lambda": 0.90,  # Slightly reduced for shorter-term advantage estimation
            "clip_range": 0.1,  # Reduced to prevent too large policy updates on small data
            "ent_coef": 0.05,  # Increased to encourage exploration in limited data
            "max_grad_norm": 0.5,  # Added to prevent explosive gradients
            "vf_coef": 0.5,  # Added to balance value function and policy learning
        }

        # Create the PPO agent
        model = PPO("MlpPolicy", env, verbose=1, **model_params)

        # Set up MLflow callback
        mlflow_callback = MLflowCallback(check_freq=10000)

        # Log parameters
        mlflow.log_params(model_params)

        # Train the agent
        model.learn(total_timesteps=120_000, callback=mlflow_callback)

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


if __name__ == "__main__":
    start()
