import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from src.utils.environment import get_env
from src.callbacks.log_test_callback import LogTestCallback, LOG_DIR
from sklearn.model_selection import TimeSeriesSplit


class BaseAgent:
    def __init__(
        self,
        experiment_name: str = "Base Agent",
        data: pd.DataFrame = [],
        model=None,
        model_kwargs: dict = {},
        callback: callable = LogTestCallback,
        env_entry_point: str = "src.environments.buy_environment.buy_environment:BuyEnvironment",
        env_kwargs: dict = {},
        train_timesteps: int = 4_500_000,
        check_freq: int = 1000,
        test_size: float = 0.1,
        n_splits: int = 1,
        num_cores: int = 1,
        main_price_column: str = "bid_price",
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
        self.num_cores = num_cores
        self.main_price_column = main_price_column

    def start(self):
        if self.model is None:
            raise ValueError("Model is required")
        if self.env_kwargs["observation"] is None:
            raise ValueError("Observation is required in env_kwargs")
        if self.env_kwargs["reward_func"] is None:  # Rename into reward in environment
            raise ValueError("Reward function is required in env_kwargs")

        self.setup_experiment()

        if mlflow.active_run() is not None:
            mlflow.end_run()

        with mlflow.start_run():
            if self.n_splits > 1:
                tscv = TimeSeriesSplit(n_splits=self.n_splits)
                tscv_split_data = tscv.split(self.data)
                for fold, (train_index, test_index) in enumerate(tscv_split_data, 1):
                    with mlflow.start_run(nested=True, run_name=f"Fold {fold}"):
                        train_data = self.data.iloc[train_index]
                        test_data = self.data.iloc[test_index]
                        self.log_data(train_data, test_data)
            else:
                train_data, test_data = train_test_split(
                    self.data, test_size=self.test_size, shuffle=False
                )
                self.log_data(train_data, test_data)

        return self.model

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
        mlflow.log_param("train_data_shape", train_data.shape)
        mlflow.log_param(
            "train_data_buy_and_hold_diff",
            train_data.iloc[-1][self.main_price_column]
            - train_data.iloc[0][self.main_price_column],
        )
        mlflow.log_param("test_data_shape", test_data.shape)
        mlflow.log_param(
            "test_data_buy_and_hold_diff",
            test_data.iloc[-1][self.main_price_column]
            - test_data.iloc[0][self.main_price_column],
        )
        for i, row in train_data.iterrows():
            mlflow.log_metric(f"training/data", row[self.main_price_column], step=i)
        for i, row in test_data.iterrows():
            mlflow.log_metric(f"test/data", row[self.main_price_column], step=i)

        # SETUP ENVIRONMENTS
        environment_name = get_environment_name(self.env_entry_point)
        base_env_kwargs = self.env_kwargs.copy()
        base_env_kwargs["start_index"] = train_data.index[0]
        base_env_kwargs["data"] = train_data
        print(f"Environment name: {environment_name}")
        print(f"Env entry point: {self.env_entry_point}")
        env = DummyVecEnv(
            [
                lambda: get_env(
                    environment_name, self.env_entry_point, LOG_DIR, base_env_kwargs
                )
                for _ in range(self.num_cores)
            ]
        )

        # SETUP TEST ENV
        # Evaluate on test data
        test_env_kwargs = self.env_kwargs.copy()
        test_env_kwargs["start_index"] = test_data.index[0]
        test_env_kwargs["data"] = test_data
        test_env = DummyVecEnv(
            [
                lambda: get_env(
                    environment_name, self.env_entry_point, LOG_DIR, test_env_kwargs
                )
            ]
        )

        # SETUP MLFLOW CALLBACK
        model_callback = self.callback(
            check_freq=self.check_freq, name="training", test_env=test_env
        )

        # SETUP MODEL
        start_index = self.env_kwargs[
            "observation"
        ].get_start_index()  # should be renamed into get_start_index
        batch_size = len(train_data) - start_index
        self.model_kwargs["batch_size"] = batch_size
        self.model_kwargs["n_steps"] = batch_size * 2
        self.model_kwargs["env"] = env
        model = self.model(**self.model_kwargs)

        # RUN TRAINING WITH MLFLOW CALLBACK, LOG ALL TEST IN MLFLOW CALLBACK
        model.learn(total_timesteps=self.train_timesteps, callback=model_callback)
        mlflow.pytorch.log_model(model, "model")


def get_environment_name(env_path: str) -> str:
    """
    Extracts the class name from the given environment path.

    Args:
        env_path (str): The full path to the environment class.

    Returns:
        str: The class name of the environment.
    """
    return env_path.split(":")[-1]
