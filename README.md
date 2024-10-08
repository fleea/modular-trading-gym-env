# Flexible Reinforcement Learning Environment for Financial Trading Simulations

This repository implements a flexible reinforcement learning (RL) environment for simulating financial trading scenarios. It is designed to facilitate experimentation with various observation and reward strategies, enabling researchers and practitioners to refine RL models for trading applications rapidly

The environment is built upon the OpenAI Gymnasium framework and leverages Stable Baselines3 for implementing RL algorithms. By modularizing key components, the repository allows for the seamless interchange of modules across different experiments, promoting reuse and scalability.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Example Usage](#example-usage)
- [Module structure](#module-structure)
  - [Folder Structure](#folder-structure)
  - [Agents](#agents)
  - [Environments](#environments)
  - [Observations](#observations) - Detailed in [Observations.md](docs/Observations.md)
  - [Rewards](#rewards)
- [Customization](#customization)
  - [Creating New Agent](#creating-new-agent)
  - [Creating New Environment](#creating-new-environment)
  - [Creating New Observation](#creating-new-observation)
  - [Creating New Reward](#creating-new-reward)
- [Testing](#testing)
- [Generate Requirements](#generate-requirements)
- [Logging](#logging)

## Features

- **Modular Architecture**: The repository decomposes the RL framework into interchangeable modules, allowing for easy customization and transferability across experiments.
- **Flexible Observations and Rewards**: Experiment with multiple observation spaces and reward strategies without altering the core environment.
- **Separation of Concerns**: Clear division of functionality into distinct modules enhances code readability and maintainability.
- **Compatibility**: Built on top of OpenAI Gymnasium and Stable Baselines3 for robust RL algorithm support.
- **Logging with MLflow**: Integrated logging and model saving using MLflow for experiment tracking and reproducibility.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/modular-trading-gym-env.git
   cd modular-trading-gym-env
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Example Usage

To run an experiment, run a file in `src/agents/`, for example:

```sh
export PYTHONPATH=$PYTHONPATH:. # Make sure to set the PYTHONPATH on root
python3 src/agents/multiple_buy_agent.py # Run an agent
python3 src/agents/single_buy_agent.py
python3 src/agents/multiple_buy_agent_expanding_window.py
```

## Module structure

The repository is organized into modular components, each responsible for a specific aspect of the RL environment. This design follows the principles of separation of concerns and uses Python Protocols for defining minimal structure requirements per component.

### Folder Structure

- `src/`: Main source code directory
  - **`agents/`**: RL agent implementations
  - `callbacks/`: Logging, testing and saving models on training
  - `data/`: Raw data (tick, news, stocks, etc)
  - `enums/`: Enumeration classes (will be deprecated)
  - **`environments/`**: Trading environment implementations
  - `interfaces/`: Interface definitions
  - **`observations/`**: Observation space definitions
  - `preprocessing/`: Data preprocessing pipelines
  - **`rewards/`**: Reward function implementations
  - `utils/`: Utility functions and helpers
  - `visualization/`: Utilities to display raw or processed data before training

### Agents (`src/agents/`) <a id="agents"></a>

The entry point for running experiments. It defines configurations for the agent, including:

- Which RL algorithm to use,
- Hyperparameters,
- Which environment to use,
- Which reward function to use,
- Which observation to use,
- How many splits to use for testing,
- How many timesteps to train for.

### Observations (`src/observations/`) <a id="observations"></a>

Define how the environment's state is represented to the agent. **Feature engineering** is performed here. See [Observations.md](docs/Observations.md) for more details about protocol and defaultimplementation.

### Rewards (`src/rewards/`) <a id="rewards"></a>

Calculate the reward strategies for the agent. Should contains a function that accept the environment as an argument and return a single float.

Some default reward functions:

- `simple_reward.py`: Simple reward based on the equity value
- `non_zero_buy_reward.py`
- `non_zero_buy_reward_stock.py`

### Environments (`src/environments/`) <a id="environments"></a>

Provides core functionality for trading simulations, including:

- Order management
- Account state updates (balance, holdings)
- Market simulation logic

BaseEnvironment is a Protocol class that defines the minimal interface for an environment. It requires several methods to be implemented:

- `step()`: Take a step in the environment
- `get_observation()`: Get the observation
- `get_current_price()`: Get the current price of the asset

Some optional methods that has default implementation:

- `reset()` (optional, has default implementation): Reset the environment to the initial state
- `_get_info()`

## Customization

### Creating a New Agent <a id="creating-new-agent"></a>

1. Create a new file in `src/agents/`
2. Implement a new class that inherits from `BaseAgent`, pick reward, observation and environment for the agent by setting the corresponding parameters

   ```
   agent = BaseAgent(
      experiment_name=experiment_name,
      data=data, # data from data/, can be preprocessed in preprocessing/ or raw
      model=RecurrentPPO, # model from stable_baselines3 (or any other sb3 compatible model)
      model_kwargs=model_kwargs, # model specific parameters
      callback=LogTestLSTMCallback, # callback from callbacks/
      env_entry_point="src.environments.buy_environment.buy_environment:BuyEnvironment", # environment from environments/
      env_kwargs=env_kwargs, # environment specific parameters
      train_timesteps=train_timesteps, # number of training timesteps
      check_freq=1000, # number of timesteps between testing
      test_size=0.1, # size of the test set
      n_splits=1, # number of splits for testing
   )
   ```

3. Run the experiment with `agent.start()`

### Creating a New Environment <a id="creating-new-environment"></a>

1. Create a new file in `src/environments/`
2. Implement a new class that inherits from `BaseEnvironment`
3. Override the necessary methods like `step()`, `reset()`, etc.

### Implementing a New Observation <a id="creating-new-observation"></a>

1. Create a new file in `src/observations/`
2. Implement a new class that inherits from `BaseObservation`
3. Implement the required methods: `get_space()`, `get_observation()`
4. Functions like `get_min_periods()` have default implementations in the base class, but override-able

### Creating a New Reward Function <a id="creating-new-reward"></a>

1. Add a new function in `src/rewards/`
2. The function should take the environment as an argument and return a float

## Testing

To run the tests:

```
pytest
```

## Generate requirements automatically <a id="generate-requirements"></a>

```bash
pip3 install pipreqs
pip3 install pip-tools
pipreqs --savepath=requirements.in && pip-compile
```

## Logging and Experiment Tracking <a id="logging"></a>

This project uses MLflow for experiment tracking. To view the MLflow UI:

1. Run an experiment in `src/agents/`
2. Start the MLflow UI:
   ```
   mlflow ui
   ```
3. Open a web browser and go to `http://127.0.0.1:5000/`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
