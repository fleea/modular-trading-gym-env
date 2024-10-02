# export PYTHONPATH=$PYTHONPATH:.
# python3.12 src/agents/multiple_buy_agent_augment_hlc.py
# Need to run in the root dir so the mlruns directory is located in the root

# Same as multiple_buy_agent_mlp_stock.py but using data that has been augmented with HLC

from stable_baselines3 import PPO
from src.rewards.non_zero_buy_reward_stock import NonZeroBuyRewardStock
from src.callbacks.log_test_callback import LogTestCallback
from src.agents.base_agent import BaseAgent
import pandas as pd
from src.preprocessing.hlc import augment_with_hlc
from src.observations.hlc_observation import HLCObservation

import weakref

def start():
    experiment_name = "PPO - Stock - Preprocessing HLC"
    train_timesteps = 30_000
    df = pd.read_csv('src/data/SP_SPX_1D.csv')
    df['time'] = pd.to_datetime(df['time'])
    df['timestamp'] = df['time']
    df.set_index('time', inplace=True)
    data = augment_with_hlc(df)
    data.reset_index(drop=True, inplace=True)
    observation = HLCObservation()
    env_kwargs = {
        "initial_balance": 10_000,
        "reward_func": NonZeroBuyRewardStock,
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
        experiment_name = experiment_name,
        data = data,
        model = PPO,
        model_kwargs = model_kwargs,
        callback = LogTestCallback,
        env_entry_point = "src.environments.multiple_buy.multiple_buy_environment_stock:MultipleBuyEnvironmentStock",
        env_kwargs = env_kwargs,
        train_timesteps = train_timesteps,
        check_freq = 1000,
        test_size = 0.1,
        n_splits = 5,
        num_cores = 1,
        main_price_column = "close",
    )
    agent.start()


def has_weakref(obj):
    try:
        weakref.ref(obj)
        return True
    except TypeError:
        return False

# Example usage

if __name__ == "__main__":
    print(has_weakref(LogTestCallback))
    print(has_weakref(NonZeroBuyRewardStock))
    print(has_weakref(HLCObservation))
    start()