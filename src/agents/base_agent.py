import mlflow
import numpy as np
from typing import List
import pandas as pd
from src.observations.base_observation import BaseObservation
from src.observations.trend_observation_rms import TrendObservationRMS
from src.rewards.non_zero_buy_reward import NonZeroBuyReward
from sklearn.model_selection import train_test_split
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from src.utils.environment import get_env
from src.utils.mlflow import MLflowCallback, LOG_DIR
from sb3_contrib import RecurrentPPO
from src.callbacks.log_test_callback import LogTestCallback
from sklearn.model_selection import TimeSeriesSplit
"""
PARAMS
environment
observation class
reward function
model implementation


observation = PriceObservation(column_name='bid_price')
env_kwargs = {
    "initial_balance": initial_balance,
    "tick_data": train_data,
    "reward_func": NonZeroBuyReward,
    "observation": observation,
    "max_orders": max_order,
    "lot": lot,
    "start_index": train_data.index[0]
}
num_cores = 4  # Define the number of cores to be used
env = DummyVecEnv([lambda: get_env("MultipleBuyEnv", "src.environments.multiple_buy.multiple_buy_environment:MultipleBuyEnvironment", LOG_DIR, env_kwargs) for _ in range(num_cores)])


reward = NonZeroBuyReward
start_experiment(
    experiment_name = "Base Agent",
    train_timesteps = 4_500_000,
    model = None,
    tick_data = [],
    max_order = 1,
    initial_balance = 10_000,
    check_freq = 1000,
    test_size = 0.1,
)

reward + observation + train data --> train environment
reward + observation + test data --> test environment
"""

class BaseAgent:
    def __init__(
            self, 
            experiment_name: str = "Base Agent", 
            data: pd.DataFrame = [], 
            model = None, 
            model_kwargs: dict = {}, 
            callback: callable = LogTestCallback, 
            env_entry_point: str = "src.environments.multiple_buy.multiple_buy_environment:MultipleBuyEnvironment", 
                env_kwargs: dict = {}, 
            train_timesteps: int = 4_500_000, 
            check_freq: int = 1000, 
            test_size: float = 0.1, 
            n_splits: int = 1
        ):
        self.experiment_name = experiment_name
        self.data = data
        self.model = model
        self.model_kwargs = model_kwargs
        self.callback = callback
        self.env_entry_point = env_entry_point
        self.env_kwargs = env_kwargs
        self.train_timesteps = train_timesteps
        self.check_freq = check_freq
        self.test_size = test_size
        self.n_splits = n_splits

    def start(self):
        if self.model is None:
            raise ValueError("Model is required")
        if self.env_kwargs['observation'] is None:
            raise ValueError("Observation is required in env_kwargs")
        if self.env_kwargs['reward_func'] is None: # Rename into reward in environment
            raise ValueError("Reward function is required in env_kwargs")
        
        self.setup_experiment()
        
        if mlflow.active_run() is not None:
            mlflow.end_run()

        with mlflow.start_run():
            if(self.n_splits > 1):
                tscv = TimeSeriesSplit(n_splits=self.n_splits)
                tscv_split_data = tscv.split(self.data)
                for fold, (train_index, test_index) in enumerate(tscv_split_data, 1):
                    with mlflow.start_run(nested=True, run_name=f"Fold {fold}"):
                        train_data = self.data.iloc[train_index]
                        test_data = self.data.iloc[test_index]
                        self.log_data(train_data, test_data)
            else:
                train_data, test_data = train_test_split(self.data, test_size=self.test_size, shuffle=False)
                self.log_data(train_data, test_data) 
    
    def setup_experiment(self):
        mlflow.set_experiment(self.experiment_name)
        mlflow.log_param("entry_point", self.env_entry_point)
        mlflow.log_param("total_data_shape", self.data.shape)
        mlflow.log_param("train_timesteps", self.train_timesteps)
        mlflow.log_param("check_freq", self.check_freq)
        mlflow.log_param("test_size", self.test_size)
        mlflow.log_param("n_splits", self.n_splits)
        for key, value in self.env_kwargs.items():
            mlflow.log_param(f"env_kwargs/{key}", value)
        for key, value in self.model_kwargs.items():
            mlflow.log_param(f"model_kwargs/{key}", value)

    def log_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        # LOG TRAINING DATA AND TEST DATA
        mlflow.log_param("train_data_shape", train_data.shape)
        mlflow.log_param("train_data_buy_and_hold_diff", train_data.iloc[-1]['bid_price'] - train_data.iloc[0]['bid_price'])
        mlflow.log_param("test_data_shape", test_data.shape)
        mlflow.log_param("test_data_buy_and_hold_diff", test_data.iloc[-1]['bid_price'] - test_data.iloc[0]['bid_price'])
        for i, row in train_data.iterrows():
            mlflow.log_metric(f"training/tick_data", row["bid_price"], step=i)
        for i, row in test_data.iterrows():
            mlflow.log_metric(f"test/tick_data", row["bid_price"], step=i)

        # SETUP ENVIRONMENTS
        environment_name = get_environment_name(self.env_entry_point)
        base_env_kwargs = self.env_kwargs.copy()
        base_env_kwargs['start_index'] = train_data.index[0]
        base_env_kwargs['tick_data'] = train_data
        num_cores = 4  # Define the number of cores to be used
        print(f"Environment name: {environment_name}")
        print(f"Env entry point: {self.env_entry_point}")
        env = SubprocVecEnv([lambda: get_env(environment_name, self.env_entry_point, LOG_DIR, base_env_kwargs) for _ in range(num_cores)])

        # SETUP TEST ENV
        # Evaluate on test data
        test_env_kwargs = self.env_kwargs.copy()
        test_env_kwargs["start_index"] = test_data.index[0]
        test_env_kwargs["tick_data"] = test_data
        test_env = SubprocVecEnv([lambda: get_env(environment_name, self.env_entry_point, LOG_DIR, test_env_kwargs)])

        # SETUP MLFLOW CALLBACK
        model_callback = self.callback(check_freq=self.check_freq, name="training", test_env=test_env)
        
        # SETUP MODEL
        start_index = self.env_kwargs['observation'].get_start_padding() # should be renamed into get_start_index
        batch_size = len(train_data) - start_index
        self.model_kwargs['batch_size'] = batch_size
        self.model_kwargs['n_steps'] = batch_size * 2
        self.model_kwargs['env'] = env
        model = self.model(**self.model_kwargs)

        # RUN TRAINING WITH MLFLOW CALLBACK, LOG ALL TEST IN MLFLOW CALLBACK
        model.learn(total_timesteps=self.train_timesteps, callback=model_callback)
        
        
def get_environment_name(env_path: str) -> str:
    """
    Extracts the class name from the given environment path.

    Args:
        env_path (str): The full path to the environment class.

    Returns:
        str: The class name of the environment.
    """
    return env_path.split(":")[-1]

