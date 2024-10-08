import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.error import NameNotFound
from typing import Dict, Any
from stable_baselines3.common.monitor import Monitor


def get_env(
    env_id: str, entry_point: str, log_dir: str = None, kwargs: Dict[str, Any] = {}
) -> gym.Env:
    """
    Registers the environment if it hasn't been registered yet and returns an instance of it.

    Args:
        env_id (str): The ID to register the environment with.
        entry_point (str): The entry point for the environment class.
        log_dir (str): The log directory for the environment class.
        kwargs (Dict[str, Any]): Keyword arguments to pass to the environment constructor.

    Returns:
        gym.Env: An instance of the requested environment.
    """
    try:
        print(env_id)
        # Try to create the environment
        env = gym.make(env_id, **kwargs)
    except NameNotFound:
        # If not found, register the environment
        register(id=env_id, entry_point=entry_point, kwargs=kwargs)
        # Now create the environment
        env = gym.make(env_id, **kwargs)
    return Monitor(env, log_dir) if log_dir else env
