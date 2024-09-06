# export PYTHONPATH=$PYTHONPATH:.
# python3.12 src/agents/multiple_buy_agent_train_test.py
# Need to run in the root dir so the mlruns directory is located in the root

import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
# from src.observations.trend_observation import TrendObservation
# from src.observations.trend_observation_percentage import TrendObservationPercentage
from src.observations.trend_observation_rms import TrendObservationRMS
from src.utils.tick_data import get_data, plot_tick_data
from sklearn.model_selection import TimeSeriesSplit
from src.utils.mlflow import MLflowCallback, LOG_DIR
from src.rewards.non_zero_buy_reward import NonZeroBuyReward
from src.utils.environment import get_env
import math


def start():
    experiment_name = "PPO Trading Agent - MultipleBuyAgent Train Test"
    mlflow.set_experiment(experiment_name)

    ### CONSTANTS ###

    # Data
    data_length = 1000
    tick_data, max_profit = get_data(data_length)
    # plot_tick_data(tick_data)

    # Trend Observation
    trend_offset = [1, 2, 5]
    std_multiplier = 1
    trend_observation = TrendObservationRMS(trend_offset, std_multiplier)

    # Environment parameters
    lot = 0.01 * 100_000
    max_order = 1
    initial_balance = 10_000

    # Training parameters
    timesteps_per_fold = 200_000
    timesteps_per_eval = 1000
    fold_length = 2
    experiment_timesteps = timesteps_per_fold * fold_length
    test_size = int(len(tick_data) * 0.2)

    # MLflow callbacks
    train_check_freq = 1000
    # eval_check_freq = 100
    # eval_train_name = 'eval_train'
    # eval_test_name = 'eval_test'
    mlflow_callback = MLflowCallback(check_freq=train_check_freq)
    # mlflow_callback_eval_train = MLflowCallback(check_freq=eval_check_freq, name=eval_train_name)
    # mlflow_callback_eval_test = MLflowCallback(check_freq=eval_check_freq, name=eval_test_name)

    def log_constants():
        mlflow.log_param("data/length", data_length)
        mlflow.log_param("data/max_profit", max_profit * lot * max_order)
        mlflow.log_param("trends/std_multiplier", std_multiplier)
        mlflow.log_param("trends/trend_offset", trend_offset)
        mlflow.log_param("env/lot", lot)
        mlflow.log_param("env/max_order", max_order)
        mlflow.log_param("env/initial_balance", initial_balance)
        mlflow.log_param("experiment/total_timesteps", experiment_timesteps)
        mlflow.log_param("experiment/timesteps_per_fold", timesteps_per_fold)
        mlflow.log_param("experiment/timesteps_per_eval", timesteps_per_eval)
        mlflow.log_param("experiment/fold_length", fold_length)
        mlflow.log_param("experiment/test_size", test_size)

    tscv = TimeSeriesSplit(n_splits=fold_length, test_size=test_size)
    with mlflow.start_run():
        log_constants()

        for fold, (train_index, test_index) in enumerate(tscv.split(tick_data), 1):
            with mlflow.start_run(nested=True, run_name=f"Fold {fold}"):
                mlflow.log_param("experiment/fold_index", fold)
                log_constants()
                mlflow.log_param("experiment/train_index", train_index)
                mlflow.log_param("experiment/test_index", test_index)



                #SETUP TRAIN ENVIRONMENT
                train_data = tick_data.iloc[train_index]
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
                
                # SETUP TEST ENVIRONMENT
                test_data = tick_data.iloc[test_index]
                test_env_kwargs = env_kwargs.copy()
                test_env_kwargs["start_index"] = test_index[0]
                test_env_kwargs["tick_data"] = test_data
                test_env = DummyVecEnv([lambda: get_env("MultipleBuyEnv", "src.environments.multiple_buy.multiple_buy_environment:MultipleBuyEnvironment", LOG_DIR, test_env_kwargs)])
                

                # Log tick data
                for i, row in train_data.iterrows():
                    mlflow.log_metric(f"eval_train/tick_data", row["bid_price"], step=i)

                for i, row in test_data.iterrows():
                    mlflow.log_metric(f"eval_test/tick_data", row["bid_price"], step=i)

                # SETUP MODEL
                batch_size = len(train_data) - max(trend_offset)
                model_params = {
                    "learning_rate": 1e-3,
                    "n_steps": 2 * batch_size,
                    "batch_size": batch_size,
                    "n_epochs": 10,
                    "gamma": 0.95,
                    "gae_lambda": 0.90,
                    "clip_range": 0.1,
                    "ent_coef": 0.05,
                    "max_grad_norm": 0.5,
                    "vf_coef": 0.5,
                }
                model = PPO("MlpPolicy", env, verbose=1, **model_params)
                # eval_steps_per_fold = timesteps_per_fold / timesteps_per_eval

                mlflow_callback = MLflowCallback(check_freq=train_check_freq, test_env=test_env)
                model.learn(total_timesteps=timesteps_per_fold, callback=mlflow_callback)
                    
                # PERFORM TRAINING AND EVALUATION PER EVAL STEP
                # for _ in range(0, math.ceil(eval_steps_per_fold)):
                #     model.learn(total_timesteps=timesteps_per_eval, callback=mlflow_callback, reset_num_timesteps= False)
                #     mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=timesteps_per_eval, deterministic=True, callback=mlflow_callback_eval_train)
                #     mean_reward_test, std_reward_test = evaluate_policy(model, test_env, n_eval_episodes=timesteps_per_eval, deterministic=True, callback=mlflow_callback_eval_test)
                #     mlflow.log_metric(f"{eval_train_name}/mean_reward", mean_reward)
                #     mlflow.log_metric(f"{eval_train_name}/std_reward", std_reward)
                #     mlflow.log_metric(f"{eval_test_name}/mean_reward", mean_reward_test)
                #     mlflow.log_metric(f"{eval_test_name}/std_reward", std_reward_test)

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