# export PYTHONPATH=$PYTHONPATH:.
# python3.12 src/agents/multiple_buy_agent_lstm.py
# Need to run in the root dir so the mlruns directory is located in the root

from sb3_contrib import RecurrentPPO
from src.observations.trend_observation_rms import TrendObservationRMS
from src.rewards.non_zero_buy_reward import NonZeroBuyReward
from src.utils.tick_data import get_real_data_per_year, get_data
from src.callbacks.log_test_lstm_callback import LogTestLSTMCallback
from src.agents.base_agent import BaseAgent
import random

def start():
    experiment_name = "Recurrent PPO Trading Agent - LSTM Policy"
    train_timesteps = 4_500_000
    data, max_profit = get_data(250, 3494, 1, 3, lambda step: max(0.00001, random.gauss(1, 0.1)))
    trend_offset = [1, 2, 3]
    std_multiplier = 1
    observation = TrendObservationRMS(trend_offset, std_multiplier)
    env_kwargs = {
        "initial_balance": 10_000,
        "reward_func": NonZeroBuyReward,
        "observation": observation,
        "max_orders": 1,
        "lot": 1,
    }
    model_kwargs = {
        "policy": "MlpLstmPolicy",
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
        model = RecurrentPPO,
        model_kwargs = model_kwargs,
        callback = LogTestLSTMCallback,
        env_entry_point = "src.environments.multiple_buy.multiple_buy_environment:MultipleBuyEnvironment",
        env_kwargs = env_kwargs,
        train_timesteps = train_timesteps,
        check_freq = 1000,
        test_size = 0.1,
        n_splits = 1,
    )
    agent.start()

if __name__ == "__main__":
    start()