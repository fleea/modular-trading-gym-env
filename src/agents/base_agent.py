import mlflow
import numpy as np
from typing import List
import pandas as pd
from src.observations.base_observation import BaseObservation
from src.observations.trend_observation_rms import TrendObservationRMS
from src.rewards.non_zero_buy_reward import NonZeroBuyReward
from sklearn.model_selection import train_test_split
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from src.utils.mlflow import MLflowCallback



# This class responsible for starting and logging the experiment
def start_experiment(
        experiment_name: str = "Base Agent",
        train_timesteps: int = 4_500_000,
        tick_data: pd.DataFrame = [],
        observation: BaseObservation = TrendObservationRMS,
        reward: callable = NonZeroBuyReward,
        max_order: int = 1,
        initial_balance: float = 10_000,
        check_freq: int = 1000,
        test_size: float = 0.1,
        env: DummyVecEnv | SubprocVecEnv = None,
        test_env: DummyVecEnv | SubprocVecEnv = None,
        model = None,
):
    mlflow.set_experiment(experiment_name)
    mlflow.log_param("total_tick_data_shape", tick_data.shape)
    mlflow.log_param("train_timesteps", train_timesteps)
    mlflow.log_param("max_order", max_order)
    mlflow.log_param("initial_balance", initial_balance)

    with mlflow.start_run():
        # THIS PART SHOULD BE UNCOMMENTED WHEN WE WANT TO DO TIME SERIES SPLIT
        # Make sure to comment the train_test_split below + indent the code below it
        # tscv = TimeSeriesSplit(n_splits=2)
        # tscv_split_data = tscv.split(tick_data)
        # print(f"tscv_split_data: {tscv_split_data}")
        # for fold, (train_index, test_index) in enumerate(tscv_split_data, 1):
        #     with mlflow.start_run(nested=True, run_name=f"Fold {fold}"):
        # train_data = tick_data.iloc[train_index]
        # test_data = tick_data.iloc[test_index]

        train_data, test_data = train_test_split(tick_data, test_size=test_size, shuffle=False)

        # LOG TRAINING DATA AND TEST DATA
        mlflow.log_param("train_data_shape", train_data.shape)
        mlflow.log_param("test_data_shape", test_data.shape)
        mlflow.log_param("train_data_buy_and_hold_diff", train_data.iloc[-1]['bid_price'] - train_data.iloc[0]['bid_price'])
        mlflow.log_param("test_data_buy_and_hold_diff", test_data.iloc[-1]['bid_price'] - test_data.iloc[0]['bid_price'])
        for i, row in train_data.iterrows():
            mlflow.log_metric(f"training/tick_data", row["bid_price"], step=i)
        for i, row in test_data.iterrows():
            mlflow.log_metric(f"test/tick_data", row["bid_price"], step=i)

        # SETUP ENV, TEST_ENV, MODEL
        mlflow_callback = MLflowCallback(check_freq=check_freq, name="training", test_env=test_env)
        
        # RUN TRAINING WITH MLFLOW CALLBACK
        model.learn(total_timesteps=train_timesteps, callback=mlflow_callback)

        
        
    