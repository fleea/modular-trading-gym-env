# THIS AGENT IS NOT USING BASE_AGENT YET
# PROBABLY WILL BE DEPRECATED

# export PYTHONPATH=$PYTHONPATH:.
# python3.12 src/agents/multiple_buy_agent_expanding_window.py
# Need to run in the root dir so the mlruns directory is located in the root

import mlflow
import math as std_math
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from src.observations.trend_observation_rms import TrendObservationRMS
from src.observations.price_observation import PriceObservation
from src.rewards.non_zero_buy_reward import NonZeroBuyReward
from src.utils.environment import get_env
from src.callbacks.log_test_callback import LogTestCallback, LOG_DIR
from src.utils.tick_data import get_real_data_per_year, get_data
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from src.utils.data_cleaning import filter_noise
import numpy as np
import random

def start():
    experiment_name = "PPO Trading Agent - Expanding Window"
    mlflow.set_experiment(experiment_name)

    # ALL CONSTANTS
    train_timesteps = 4_500_000
    # n_eval_episodes = 100 # For both train and test, on evaluate_policy
    # tick_data, max_profit = get_data(1000)  # Generate 1000 data points
    # tick_data = get_real_data_per_year('src/data/SP_SPX_1D.csv', 2013, 2023)
    # tick_data = filter_noise(tick_data, 1, 'bid_price')
    tick_data, max_profit = get_data(250, 3494, 1, 3, lambda step: max(0.00001, random.gauss(1, 0.1)))
    print(f"Tick data shape: {tick_data.shape}")
    # print(f"Max profit: {max_profit}")
    # mlflow.log_param("max_profit", max_profit)
    mlflow.log_param("tick_data_shape", tick_data.shape)
    

    trend_offset = [1, 2, 3, 4, 5, 10, 40]
    std_multiplier = 1
    # trend_observation = TrendObservationRMS(trend_offset, std_multiplier)
    trend_observation = PriceObservation(column_name='bid_price')
    # lot = 0.01 * 100_000 # USD
    lot = 1
    max_order = 1
    initial_balance = 10_000
    # mlflow_callback_eval_train = MLflowCallback(check_freq=100, name="eval_train")
    # mlflow_callback_eval_test = MLflowCallback(check_freq=100, name="eval_test")
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        # mlflow.log_param("max_profit", max_profit * lot * max_order)
        mlflow.log_param("std_multiplier", std_multiplier)
        mlflow.log_param("trend_offset", trend_offset)

        # THIS PART SHOULD BE UNCOMMENTED WHEN WE WANT TO DO TIME SERIES SPLIT
        # Make sure to comment the train_test_split below + indent the code below it
        # tscv = TimeSeriesSplit(n_splits=2)
        # tscv_split_data = tscv.split(tick_data)
        # print(f"tscv_split_data: {tscv_split_data}")
        # for fold, (train_index, test_index) in enumerate(tscv_split_data, 1):
        #     with mlflow.start_run(nested=True, run_name=f"Fold {fold}"):
        # train_data = tick_data.iloc[train_index]
        # test_data = tick_data.iloc[test_index]

        train_data, test_data = train_test_split(tick_data, test_size=0.1, shuffle=False)
        mlflow.log_param("train_data_size", len(train_data))
        mlflow.log_param("test_data_size", len(test_data))
        mlflow.log_param("train_data_diff", train_data.iloc[-1]['bid_price'] - train_data.iloc[0]['bid_price'])
        mlflow.log_param("test_data_diff", test_data.iloc[-1]['bid_price'] - test_data.iloc[0]['bid_price'])
        
        # Log tick data
        for i, row in train_data.iterrows():
            mlflow.log_metric(f"eval_train/tick_data", row["bid_price"], step=i)

        for i, row in test_data.iterrows():
            mlflow.log_metric(f"eval_test/tick_data", row["bid_price"], step=i)

        env_kwargs = {
            "initial_balance": initial_balance,
            "tick_data": train_data,
            "reward_func": NonZeroBuyReward,
            "observation": trend_observation,
            "max_orders": max_order,
            "lot": lot,
            "start_index": train_data.index[0]
        }
        num_cores = 4  # Define the number of cores to be used
        env = DummyVecEnv([lambda: get_env("MultipleBuyEnv", "src.environments.multiple_buy.multiple_buy_environment:MultipleBuyEnvironment", LOG_DIR, env_kwargs) for _ in range(num_cores)])

        # LOGGING Add these debug prints
        # base_env = env.envs[0]
        # print(f"STARTING FOLD {fold}")
        # print(f"Initial current_index: {base_env.unwrapped.current_index}")
        # print(f"start_index: {base_env.unwrapped.start_index}")
        # print(f"max_index: {base_env.unwrapped.max_index}")
        # print(f"Tick data shape: {base_env.unwrapped.tick_data.shape}")
        # print(f"Tick data index: {base_env.unwrapped.tick_data.index}")
        # print(f"Train: data={train_data}")
        # print(f"Test:  data={test_data}")

        batch_size = len(train_data) - max(trend_offset)
        batch_size = std_math.floor(batch_size / 10)
        model_params = {
            "learning_rate": 1e-3,
            "n_steps": batch_size * 2,
            "batch_size": batch_size,
            "n_epochs": 10,
            "gamma": 0.95,
            "gae_lambda": 0.90,
            "clip_range": 0.1,
            "ent_coef": 0.05,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "use_sde": True,
            "sde_sample_freq": 4,
            
        }

        log_metrics_from_dict(model_params)

        # plot_tick_data(train_data)
        # model = PPO("MlpPolicy", env, verbose=1, **model_params)
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            **model_params
        )

        # SETUP TEST ENV
        # Evaluate on test data
        test_env_kwargs = env_kwargs.copy()
        test_env_kwargs["start_index"] = test_data.index[0]
        test_env_kwargs["data"] = test_data
        test_env = SubprocVecEnv([lambda: get_env("MultipleBuyEnv", "src.environments.multiple_buy.multiple_buy_environment:MultipleBuyEnvironment", LOG_DIR, test_env_kwargs)])

        # SETUP MLFLOW CALLBACK WITH TEST ENV
        mlflow_callback = LogTestCallback(check_freq=1000, name="training", test_env=test_env)
        
        # RUN TRAINING WITH MLFLOW CALLBACK
        model.learn(total_timesteps=train_timesteps, callback=mlflow_callback)

        # Evaluate on train data
        # mean_reward_train, std_reward_train = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, callback=mlflow_callback_eval_train)
        # mlflow.log_metric("train/mean_reward", mean_reward_train)
        # mlflow.log_metric("train/std_reward", std_reward_train)

        
        # mean_reward_test, std_reward_test = evaluate_policy(model, test_env, n_eval_episodes=n_eval_episodes, callback=mlflow_callback_eval_test)
        # mlflow.log_metric("test/mean_reward", mean_reward_test)
        # mlflow.log_metric("test/std_reward", std_reward_test)

        obs = test_env.reset()
        equity = []
        balance = []
        actions = []
        lstm_states = None

        # It is particularly important to pass the lstm_states and episode_start argument to the predict() method, so the cell and hidden states of the LSTM are correctly updated.
        num_envs = 1
        # Episode start signals are used to reset the lstm states
        episode_starts = np.ones((num_envs,), dtype=bool)
        for _ in range(len(test_data)):
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, rewards, dones, infos = test_env.step(action)
            equity.append(infos[0]['equity'])
            balance.append(infos[0]['balance'])
            actions.append(action)

            if dones.any():
                break

        # Log actions that are not "do nothing" (assuming "do nothing" action is represented by 0)
        non_zero_actions = [act for act in actions if act != 0]
        mlflow.log_metric("eval_test/actions", len(non_zero_actions))

        # Log equity movement
        for i, eq in enumerate(equity):
            mlflow.log_metric("eval_test/equity", eq, step=i)

        mlflow.log_metric("eval_test/final_equity", equity[-1])
        mlflow.log_metric("eval_test/final_balance", balance[-1])
        
def log_metrics_from_dict(metrics_dict):
    for key, value in metrics_dict.items():
        mlflow.log_param(key, value)

if __name__ == "__main__":
    start()