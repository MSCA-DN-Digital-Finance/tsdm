import pytest
import numpy as np
from tsdm.agents import (
    AlwaysUpAgent,
    RepeatLastMovementAgent,
    FrequencyBasedMajorityAgent,
    StaticMeanReversionAgent,
    DynamicMeanReversionAgent,
    SGDClassifierAgent,
    DQNAgent
)

# ------------------------------------------------------------------------
# SECTION 1 — Heuristic Baseline Agents
# ------------------------------------------------------------------------

def test_always_up_agent():
    """
    AlwaysUpAgent should always return 1 regardless of observations.
    """
    agent = AlwaysUpAgent()
    agent.observe(1.0)
    assert agent.place_bet() == 1, "AlwaysUpAgent should always bet 1"


def test_repeat_last_movement_agent():
    """
    RepeatLastMovementAgent should bet in the direction of the last movement.
    - Up after increase.
    - Down after decrease.
    """
    agent = RepeatLastMovementAgent()
    agent.observe(1.0)
    agent.observe(2.0)  # Upward movement
    assert agent.place_bet() == 1, "Should bet up after upward movement"
    agent.observe(1.5)  # Downward movement
    assert agent.place_bet() == 0, "Should bet down after downward movement"


def test_frequency_based_majority_agent():
    """
    FrequencyBasedMajorityAgent should make a bet after observing a sequence.
    Bet should always be 0 or 1.
    """
    agent = FrequencyBasedMajorityAgent()
    for val in [1.0, 2.0, 3.0, 2.5, 2.0]:
        agent.observe(val)
    assert agent.place_bet() in [0, 1], "Bet should be 0 or 1"


def test_static_mean_reversion_agent():
    """
    StaticMeanReversionAgent bets based on historical mean.
    Bet should be valid after multiple observations.
    """
    agent = StaticMeanReversionAgent()
    for val in [1.0, 2.0, 3.0, 2.5, 2.0]:
        agent.observe(val)
    assert agent.place_bet() in [0, 1], "Bet should be 0 or 1"


def test_dynamic_mean_reversion_agent():
    """
    DynamicMeanReversionAgent uses a rolling window.
    Should provide a valid bet once enough observations are made.
    """
    agent = DynamicMeanReversionAgent(time_window=3)
    for val in [1.0, 2.0, 3.0, 2.5, 2.0]:
        agent.observe(val)
    assert agent.place_bet() in [0, 1], "Bet should be 0 or 1"


# ------------------------------------------------------------------------
# SECTION 2 — Machine Learning Agents (SGDClassifier & DQN)
# ------------------------------------------------------------------------

def test_sgd_classifier_agent_prediction():
    """
    SGDClassifierAgent trains online on observed data.
    - Should predict valid action after training.
    - Should reset correctly.
    """
    agent = SGDClassifierAgent(window_size=3)
    agent.observe(1.0)
    agent.observe(2.0)
    agent.observe(3.0)
    agent.observe(4.0)  # Training should happen here

    assert agent.place_bet() in [0, 1], "SGDClassifierAgent should bet 0 or 1"

    agent.reset()
    assert agent.observed_values == [], "SGDClassifierAgent should clear observed values on reset"


def test_dqn_agent_prediction_and_training():
    """
    DQNAgent uses experience replay to learn from past observations.
    - Should produce valid bets.
    - Epsilon should decay after training.
    - Reset should clear state.
    """
    agent = DQNAgent(state_size=3)
    agent.observe(1.0)
    agent.observe(2.0)
    agent.observe(3.0)
    agent.observe(4.0)  # Enough to start storing transitions

    bet = agent.place_bet()
    assert bet in [0, 1], "DQNAgent should bet 0 or 1"

    old_epsilon = agent.epsilon
    for _ in range(100):
        agent.observe(np.random.rand())
    assert agent.epsilon <= old_epsilon, "DQNAgent epsilon should decay after training"

    agent.reset()
    assert agent.observed_values == [], "DQNAgent should clear observed values on reset"



# ------------------------------------------------------------------------
# SECTION 3 — Integration Test: All Agents on Same Dataset
# ------------------------------------------------------------------------

@pytest.mark.parametrize("agent_class", [
    AlwaysUpAgent,
    RepeatLastMovementAgent,
    FrequencyBasedMajorityAgent,
    StaticMeanReversionAgent,
    lambda: DynamicMeanReversionAgent(time_window=3),
    lambda: SGDClassifierAgent(window_size=3),
    lambda: DQNAgent(state_size=3),
])
def test_agents_on_same_random_walk(agent_class):
    """
    Integration Test:
    - Runs all agents on a shared random walk.
    - Ensures each agent:
        * Produces valid bets.
        * Can observe data without errors.
        * Resets correctly.
    """
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(100))  # Random walk of length 100

    # Instantiate agents with or without parameters
    agent = agent_class() if callable(agent_class) else agent_class

    for price in prices:
        bet = agent.place_bet()
        assert bet in [0, 1], f"{agent.__class__.__name__} bet must be 0 or 1"
        agent.observe(price)

    agent.reset()
    assert agent.observed_values == [], f"{agent.__class__.__name__} should clear observed values on reset"
