# export PYTHONPATH=$PYTHONPATH:.
# python3.12 src/agents/buy_agent_mlp_stock.py
# Need to run in the root dir so the mlruns directory is located in the root

from stable_baselines3 import PPO
from src.observations.trend_observation_percentage_array_stock import (
    TrendObservationPercentageArrayStock,
)
from src.rewards.non_zero_buy_reward import NonZeroBuyReward
from src.utils.tick_data import get_real_data_per_year
from src.callbacks.log_test_callback import LogTestCallback
from src.agents.base_agent import BaseAgent
from src.utils.data_cleaning import filter_noise


def start():
    experiment_name = "PPO - MLP Stock"
    train_timesteps = 2_000_000
    data = get_real_data_per_year("src/data/SP_SPX_1D.csv", 2013, 2023)
    data = filter_noise(data, 1, "close")
    trend_offset = [1, 2, 3, 10, 40]
    observation = TrendObservationPercentageArrayStock(trend_offset)
    env_kwargs = {
        "initial_balance": 10_000,
        "reward_func": NonZeroBuyReward,
        "observation": observation,
        "spread": 0.1,
        "max_orders": 1,
        "lot": 1,
    }
    model_kwargs = {
        "policy": "MlpPolicy",
        "verbose": 1,
        "learning_rate": 1e-3,
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
    agent = BaseAgent(
        experiment_name=experiment_name,
        data=data,
        model=PPO,
        model_kwargs=model_kwargs,
        callback=LogTestCallback,
        env_entry_point="src.environments.buy_environment_stock.buy_environment_stock:BuyEnvironmentStock",
        env_kwargs=env_kwargs,
        train_timesteps=train_timesteps,
        check_freq=1000,
        test_size=0.1,
        n_splits=1,
        num_cores=4,
    )
    agent.start()


if __name__ == "__main__":
    start()
