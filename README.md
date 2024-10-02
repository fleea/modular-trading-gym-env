# Modular Trading Gym Environment

This project implements a flexible reinforcement learning environment for simulating financial trading scenarios. It's based on the OpenAI Gymnasium framework and designed to be modular and extensible.

## Features

- Multiple environment types that can be extended and customized
- Customizable observations and reward functions
- Integration with Stable Baselines3 for easy agent implementation
- MLflow integration for experiment tracking
- Mock tick data generation utilities for testing different market conditions
- Example of agent implementation is located in `src/agents` folder

## Design Principles on Separation of Concerns

- **Lean Base Environment**: A streamlined BaseEnvironment class provides core functionality for trading simulations, handling essential operations like order management and account state updates.
- **Pluggable Reward Functions**: Reward functions are easily interchangeable, allowing rapid experimentation with different reward strategies without modifying the environment code.
- **Extensible Observation Space**: Observation types are defined through a separate BaseObservation class, enabling easy creation and swapping of different state representations.
- **Action Space** is tied into the environment (step function), changing action space requires changing step function. Different action space requires different environment implementation.

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
   pip install mlflow
   ```

## Project Structure

- `src/`: Main source code directory
  - `agents/`: RL agent implementations
  - `environments/`: Trading environment implementations
  - `observations/`: Observation space definitions
  - `rewards/`: Reward function implementations
  - `utils/`: Utility functions and helpers
  - `enums/`: Enumeration classes
  - `interfaces/`: Interface definitions

## Example Usage

To run a training session with the PPO agent on the MultipleBuyGear environment:

```sh
export PYTHONPATH=$PYTHONPATH:. # Make sure to set the PYTHONPATH on root
python3 src/agents/multiple_buy_agent.py # Run an agent
python3 src/agents/single_buy_agent.py
python3 src/agents/multiple_buy_agent_expanding_window.py
```

Tick data is still mock, feel free to load your own

## Customization

### Creating a New Environment

1. Create a new file in `src/environments/`
2. Implement a new class that inherits from `BaseEnvironment`
3. Override the necessary methods like `step()`, `reset()`, etc.

### Implementing a New Observation

1. Create a new file in `src/observations/`
2. Implement a new class that inherits from `BaseObservation`
3. Implement the required methods: `get_space()`, `get_observation()`
4. Functions like `get_min_periods()` have default implementations in the base class, but override-able

### Creating a New Reward Function

1. Add a new function in `src/rewards/`
2. The function should take the environment as an argument and return a float

## Testing

To run the tests:

```
pytest
```

## Generate requirements automatically

```bash
pip3 install pipreqs
pip3 install pip-tools
pipreqs --savepath=requirements.in && pip-compile
```

## Logging and Experiment Tracking

This project uses MLflow for experiment tracking. To view the MLflow UI:

1. Run an experiment
2. Start the MLflow UI:
   ```
   mlflow ui
   ```
3. Open a web browser and go to `http://localhost:5000`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
