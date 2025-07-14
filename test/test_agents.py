import pytest
from src.agents import (
    ZeroAgent,
    RepeatLastMovementAgent,
    FrequencyBasedMajorityAgent,
    StaticMeanReversionAgent,
    DynamicMeanReversionAgent,
    SGDClassifierTradingAgent
)


def test_zero_agent():
    agent = ZeroAgent()
    agent.observe(1.0)
    assert agent.place_bet() == 1, "ZeroAgent should always bet 1"


def test_repeat_last_movement_agent():
    agent = RepeatLastMovementAgent()
    agent.observe(1.0)
    agent.observe(2.0)  # Increase
    assert agent.place_bet() == 1, "Should bet up after upward movement"
    agent.observe(1.5)  # Decrease
    assert agent.place_bet() == 0, "Should bet down after downward movement"


def test_frequency_based_majority_agent():
    agent = FrequencyBasedMajorityAgent()
    for val in [1.0, 2.0, 3.0, 2.5, 2.0]:
        agent.observe(val)
    assert agent.place_bet() in [0, 1], "Bet should be 0 or 1"


def test_static_mean_reversion_agent():
    agent = StaticMeanReversionAgent()
    for val in [1.0, 2.0, 3.0, 2.5, 2.0]:
        agent.observe(val)
    assert agent.place_bet() in [0, 1], "Bet should be 0 or 1"


def test_dynamic_mean_reversion_agent():
    agent = DynamicMeanReversionAgent(time_window=3)
    for val in [1.0, 2.0, 3.0, 2.5, 2.0]:
        agent.observe(val)
    assert agent.place_bet() in [0, 1], "Bet should be 0 or 1"


def test_sgd_classifier_trading_agent():
    agent = SGDClassifierTradingAgent(window_size=3)
    for val in [1.0, 2.0, 3.0, 2.5, 2.0, 3.0]:
        agent.observe(val)
    assert agent.place_bet() in [0, 1], "Bet should be 0 or 1"
