import numpy as np
import mlflow
from stable_baselines3.common.callbacks import BaseCallback


LOG_DIR="./mlflow"


class MLflowCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose: int = 1):
        super(MLflowCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.current_episode_reward = 0
        self.all_episode_equities = []

    def _on_step(self) -> bool:
        try:
            for info in self.locals["infos"]:
                # Log reward for each step
                if "reward" in info:
                    reward = info["reward"]
                    self.current_episode_reward += reward
                    mlflow.log_metric("step_reward", reward, step=self.num_timesteps)

                # Log final equity and cumulative reward when episode ends
                if info.get("final_equity") is not None:
                    final_equity = info["final_equity"]
                    self.all_episode_equities.append(final_equity)

                    mlflow.log_metric(
                        "episode_final_equity", final_equity, step=self.num_timesteps
                    )
                    mlflow.log_metric(
                        "episode_cumulative_reward",
                        self.current_episode_reward,
                        step=self.num_timesteps,
                    )

                    # Reset for next episode
                    self.current_episode_reward = 0

            if self.n_calls % self.check_freq == 0 and self.all_episode_equities:
                mean_equity = np.mean(self.all_episode_equities[-100:])
                mlflow.log_metric(
                    "mean_final_equity", mean_equity, step=self.num_timesteps
                )

        except Exception as e:
            print(f"Warning: Failed to log metrics to MLflow. Error: {e}")

        return True

    def on_training_end(self) -> None:
        try:
            # Log final metrics
            if self.all_episode_equities:
                final_mean_equity = np.mean(self.all_episode_equities[-100:])
                mlflow.log_metric("final_mean_equity", final_mean_equity)
                print(f"Final Mean Equity (last 100 episodes): {final_mean_equity}")
        except Exception as e:
            print(f"Warning: Failed to log final metrics to MLflow. Error: {e}")
