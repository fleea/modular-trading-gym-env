import numpy as np
import mlflow
from stable_baselines3.common.callbacks import BaseCallback


LOG_DIR = "./mlflow"


class LogTestCallback(BaseCallback):
    def __init__(
        self,
        check_freq: int = 1000,
        name: str = "training",
        verbose: int = 1,
        test_env=None,
    ):
        super(LogTestCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.current_episode_reward = 0
        self.all_episode_equities = []
        self.name = name
        self.step_count = 0
        self.test_env = test_env

    def log_locals(self, name: str, infos: list, num_timesteps: int):
        try:
            for info in infos:
                # Log reward for each step
                if "reward" in info:
                    reward = info["reward"]
                    self.current_episode_reward += reward
                    mlflow.log_metric(f"{name}/step_reward", reward, step=num_timesteps)

                # Log final equity and cumulative reward when episode ends
                if info.get("final_equity") is not None:
                    final_equity = info["final_equity"]
                    self.all_episode_equities.append(final_equity)

                    mlflow.log_metric(
                        f"{name}/episode_final_equity", final_equity, step=num_timesteps
                    )
                    mlflow.log_metric(
                        f"{name}/episode_cumulative_reward",
                        self.current_episode_reward,
                        step=num_timesteps,
                    )

                    # Reset for next episode
                    self.current_episode_reward = 0

            if self.n_calls % self.check_freq == 0 and self.all_episode_equities:
                mean_equity = np.mean(self.all_episode_equities[-100:])
                mlflow.log_metric(
                    f"{name}/mean_final_equity", mean_equity, step=num_timesteps
                )

        except Exception as e:
            print(f"Warning: Failed to log metrics to MLflow. Error: {e}")
            print(self.locals)

    def evaluate_model(self):
        model = self.model

        # mean_reward, std_reward = evaluate_policy(model, self.test_env, n_eval_episodes=timesteps_per_eval, deterministic=True)
        # mlflow.log_metric(f"eval_train/mean_reward", mean_reward, step=self.num_timesteps)
        # mlflow.log_metric(f"eval_train/std_reward", std_reward, step=self.num_timesteps)

        # MANUAL EVALUATION
        if self.test_env is not None:
            test_env = self.test_env
            obs = test_env.reset()
            lstm_states = None
            num_envs = 1
            # Episode start signals are used to reset the lstm states
            episode_starts = np.ones((num_envs,), dtype=bool)
            while True:
                action, _states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                obs, rewards, dones, infos = test_env.step(action)

                if dones.any():
                    info = infos[0]
                    mlflow.log_metric(
                        f"eval_test/final_equity",
                        info["final_equity"],
                        step=self.num_timesteps,
                    )
                    break

    def _on_step(self) -> bool:
        """
        This method is called at each step of the training process.
        """
        self.log_locals(self.name, self.locals["infos"], self.num_timesteps)
        if self.n_calls % self.check_freq == 0:
            self.evaluate_model()
        return True

    def __call__(self, locals_, globals_):
        """
        This method is called at each call of the callback.
        """
        self.log_locals(self.name, locals_["infos"], self.step_count)
        self.step_count += 1
        return False
