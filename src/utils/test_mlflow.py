import pytest
from unittest.mock import patch
from src.utils.mlflow import MLflowCallback  # Adjust this import path as necessary


@pytest.fixture
def callback():
    return MLflowCallback(check_freq=10)


@patch("src.utils.mlflow.mlflow")
def test_on_step_logging(mock_mlflow, callback):
    # Mock the locals dictionary that would be passed by the RL algorithm
    mock_locals = {
        "infos": [
            {"reward": 1.0},
            {"reward": 2.0, "final_equity": 1000.0},
        ]
    }
    callback.locals = mock_locals
    callback.num_timesteps = 100

    # Call _on_step
    callback._on_step()

    # Check if metrics were logged correctly
    mock_mlflow.log_metric.assert_any_call("step_reward", 1.0, step=100)
    mock_mlflow.log_metric.assert_any_call("step_reward", 2.0, step=100)
    mock_mlflow.log_metric.assert_any_call("episode_final_equity", 1000.0, step=100)
    mock_mlflow.log_metric.assert_any_call("episode_cumulative_reward", 3.0, step=100)


@patch("src.utils.mlflow.mlflow")
def test_on_step_mean_equity_logging(mock_mlflow, callback):
    callback.all_episode_equities = [900.0] * 99 + [1000.0]
    callback.n_calls = 10  # Trigger check_freq condition
    callback.num_timesteps = 100
    callback.locals = {"infos": [{}]}  # Empty info to avoid processing in the main loop

    callback._on_step()

    mock_mlflow.log_metric.assert_any_call("mean_final_equity", 901.0, step=100)


@patch("src.utils.mlflow.mlflow")
def test_on_training_end(mock_mlflow, callback):
    callback.all_episode_equities = [900.0] * 99 + [1000.0]

    callback.on_training_end()

    mock_mlflow.log_metric.assert_called_once_with("final_mean_equity", 901.0)


@patch("src.utils.mlflow.mlflow")
def test_error_handling(mock_mlflow, callback):
    mock_mlflow.log_metric.side_effect = Exception("MLflow error")
    callback.locals = {"infos": [{"reward": 1.0}]}
    callback.num_timesteps = 100

    # This should not raise an exception
    callback._on_step()


@patch("src.utils.mlflow.mlflow")
def test_multiple_episodes(mock_mlflow, callback):
    # Simulate multiple episodes
    for i in range(3):
        callback.locals = {
            "infos": [
                {"reward": 1.0},
                {"reward": 2.0},
                {"reward": 3.0, "final_equity": 1000.0 + i * 100},
            ]
        }
        callback.num_timesteps = 100 + i * 10
        callback._on_step()

    # Check if episode rewards are reset correctly
    assert callback.current_episode_reward == 0
    assert len(callback.all_episode_equities) == 3
    assert callback.all_episode_equities == [1000.0, 1100.0, 1200.0]


@patch("src.utils.mlflow.mlflow")
def test_check_freq_behavior(mock_mlflow, callback):
    callback.check_freq = 3 # episode_cumulative_reward and mean_final_equity should be called every 3 step
    callback.all_episode_equities = [1000.0, 1100.0, 1200.0]
    callback.num_timesteps = 100
    callback.n_calls = 0  # Initialize n_calls

    # Helper function to simulate a step with a given reward
    def simulate_step(reward, final_equity=None):
        info = {"reward": reward}
        if final_equity is not None:
            info["final_equity"] = final_equity
        callback.locals = {"infos": [info]}
        callback._on_step()

    # Should not log mean equity, but will log step reward and cumulative reward
    simulate_step(10)  # n_calls = 1
    assert (
        mock_mlflow.log_metric.call_count == 2
    )  # step_reward, episode_cumulative_reward
    simulate_step(20)  # n_calls = 2
    assert mock_mlflow.log_metric.call_count == 4  # 2 more calls

    # Reset mock before check_freq step
    mock_mlflow.log_metric.reset_mock()

    # Should log mean equity in addition to other metrics
    simulate_step(30, final_equity=1300.0)  # n_calls = 3
    assert mock_mlflow.log_metric.call_count == 4
    mock_mlflow.log_metric.assert_any_call("step_reward", 30, step=100)
    mock_mlflow.log_metric.assert_any_call("episode_cumulative_reward", 60, step=100)
    mock_mlflow.log_metric.assert_any_call("episode_final_equity", 1300.0, step=100)
    mock_mlflow.log_metric.assert_any_call("mean_final_equity", 1150.0, step=100)

    # Reset mock and check next cycle
    mock_mlflow.log_metric.reset_mock()
    simulate_step(40)  # n_calls = 4
    simulate_step(50)  # n_calls = 5
    assert mock_mlflow.log_metric.call_count == 4  # 2 calls per step

    # Should log mean equity again
    mock_mlflow.log_metric.reset_mock()
    simulate_step(60, final_equity=1400.0)  # n_calls = 6
    assert mock_mlflow.log_metric.call_count == 4
    mock_mlflow.log_metric.assert_any_call("step_reward", 60, step=100)
    mock_mlflow.log_metric.assert_any_call("episode_cumulative_reward", 150, step=100)
    mock_mlflow.log_metric.assert_any_call("episode_final_equity", 1400.0, step=100)
    mock_mlflow.log_metric.assert_any_call("mean_final_equity", 1200.0, step=100)

    # Should log mean equity again
    mock_mlflow.log_metric.reset_mock()
    simulate_step(60)  # n_calls = 6
    assert mock_mlflow.log_metric.call_count == 2
    mock_mlflow.log_metric.assert_any_call("step_reward", 60, step=100)
    # mock_mlflow.log_metric.assert_any_call("episode_cumulative_reward", 120, step=100)
    # mock_mlflow.log_metric.assert_any_call("mean_final_equity", 1100.0, step=100)


if __name__ == "__main__":
    pytest.main()
