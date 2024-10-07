from .log_test_callback import LogTestCallback
import mlflow
import numpy as np


class LogTestLSTMCallback(LogTestCallback):
    def __init__(
        self,
        check_freq: int = 1000,
        name: str = "training",
        verbose: int = 1,
        test_env=None,
    ):
        super().__init__(check_freq, name, verbose, test_env)

    def evaluate_model(self):
        model = self.model

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
                lstm_states = _states
                obs, rewards, dones, infos = test_env.step(action)

                if dones.any():
                    info = infos[0]
                    mlflow.log_metric(
                        f"eval_test/final_equity",
                        info["final_equity"],
                        step=self.num_timesteps,
                    )
                    break
