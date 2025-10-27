import pytest
from tsdm.tasks import PredictionTask, BettingGame

# ------------------------------------------------------------------------
# SECTION 1 — Deprecation: BettingGame → PredictionTask
# ------------------------------------------------------------------------

def test_betting_game_deprecation():
    """
    Ensure the old class name `BettingGame` still works but:
    - Emits a DeprecationWarning on construction.
    - Returns an instance that is actually a `PredictionTask` (wrapper/alias).
    """

    class DummyGen:
        def generate_value(self, last):
            return last + 1

    class DummyAgent:
        def observe(self, v): 
            pass
        def place_bet(self): 
            return 1

    # Expect a DeprecationWarning when instantiating the deprecated class.
    with pytest.warns(DeprecationWarning):
        game = BettingGame(DummyGen(), DummyAgent(), total_movements=3)
        # The deprecated class should behave like the new class.
        assert isinstance(game, PredictionTask), "BettingGame should wrap PredictionTask"


# ------------------------------------------------------------------------
# SECTION 2 — PredictionTask: Core Reward Behavior (Parametrized)
# ------------------------------------------------------------------------

# --- Dummy Generators ----------------------------------------------------

class UpGenerator:
    """
    Monotonic increasing generator:
    Each call returns the next integer: 1, 2, 3, ...
    Useful to test that 'always up' agents accumulate positive reward.
    """
    def __init__(self):
        self.v = 0
    def generate_value(self, last):
        self.v += 1
        return self.v


class DownGenerator:
    """
    Monotonic decreasing generator:
    Each call returns the previous integer: -1, -2, -3, ...
    Useful to test that 'always down' agents accumulate positive reward.
    """
    def __init__(self):
        self.v = 0
    def generate_value(self, last):
        self.v -= 1
        return self.v


# --- Dummy Agents --------------------------------------------------------

class AlwaysUpAgent:
    """Always predicts 'up' (1)."""
    def observe(self, v): 
        pass
    def place_bet(self): 
        return 1


class AlwaysDownAgent:
    """Always predicts 'down' (0)."""
    def observe(self, v): 
        pass
    def place_bet(self): 
        return 0


@pytest.mark.parametrize(
    "generator, agent, total_steps, expected_reward",
    [
        # Rising series → 'up' agent is always correct; 'down' agent always wrong
        (UpGenerator(),   AlwaysUpAgent(),   5,  +5),
        (UpGenerator(),   AlwaysDownAgent(), 5,  -5),

        # Falling series → 'down' agent is always correct; 'up' agent always wrong
        (DownGenerator(), AlwaysDownAgent(), 5,  +5),
        (DownGenerator(), AlwaysUpAgent(),   5,  -5),
    ]
)
def test_prediction_task_reward_behavior(generator, agent, total_steps, expected_reward):
    """
    Parametrized reward correctness:
    - Runs the same test over multiple (generator, agent) combinations.
    - Verifies the final cumulative reward matches the expected sign and magnitude.
    """
    task = PredictionTask(generator, agent, total_steps)
    reward = task.play_game()
    assert reward == expected_reward, (
        f"Expected reward {expected_reward} but got {reward} "
        f"for {agent.__class__.__name__} on {generator.__class__.__name__}"
    )

import pytest
from tsdm.tasks import PredictionTask, BettingGame

# ------------------------------------------------------------------------
# SECTION 3 — PredictionTask: Observation & State Flow
# ------------------------------------------------------------------------

def test_prediction_task_observe_called_with_last_value_each_step():
    """
    The agent must observe the *previous* value (last_value) before each turn.
    This test records observed values and checks the exact sequence.
    """
    class IncrementGen:
        # Uses last_value to produce next value deterministically: last + 1
        def generate_value(self, last):
            return last + 1

    class RecordingAgent:
        def __init__(self):
            self.observed = []
        def observe(self, v):
            self.observed.append(v)
        def place_bet(self):
            return 1  # always up

    task = PredictionTask(IncrementGen(), RecordingAgent(), total_movements=3, start_value=10)
    task.play_game()

    # Expect to have observed the last_value before each step: 10, 11, 12
    assert task.agent.observed == [10, 11, 12], "Agent should observe last_value before each turn"


def test_prediction_task_last_value_updates_each_step():
    """
    After each step, last_value should update to the newly generated value.
    """
    class IncrementGen:
        def generate_value(self, last): return last + 2

    class UpAgent:
        def observe(self, v): pass
        def place_bet(self): return 1

    task = PredictionTask(IncrementGen(), UpAgent(), total_movements=2, start_value=0)
    task.play_game()
    # Sequence: start=0 → step1 value=2 → step2 value=4 → final last_value=4
    assert task.last_value == 4, "last_value must equal the final generated value"


# ------------------------------------------------------------------------
# SECTION 4 — PredictionTask: Logging & Development Tracking
# ------------------------------------------------------------------------

def test_prediction_task_reward_development_length_and_values():
    """
    reward_development should:
    - have one entry per step
    - reflect cumulative reward correctly
    Using a strictly increasing generator + always-up agent → [1, 2, 3, 4]
    """
    class UpGen:
        def __init__(self): self.v = 0
        def generate_value(self, last):
            self.v += 1
            return self.v

    class AlwaysUp:
        def observe(self, v): pass
        def place_bet(self): return 1

    task = PredictionTask(UpGen(), AlwaysUp(), total_movements=4, start_value=0)
    task.play_game()
    assert len(task.reward_development) == 4, "Must log one cumulative entry per step"
    assert list(task.reward_development) == [1, 2, 3, 4], "Cumulative reward should increase by +1 each step"


def test_prediction_task_bet_log_structure_and_step_indices():
    """
    bet_log entries: (step, value, bet, received_reward)
    - Should have length == total steps
    - step should start at 1 and increase by 1
    - value should be numeric
    - bet should be 0 or 1
    - received_reward should be +1 or -1
    """
    class UpGen:
        def __init__(self): self.v = 0
        def generate_value(self, last):
            self.v += 1
            return self.v

    class AlwaysUp:
        def observe(self, v): pass
        def place_bet(self): return 1

    task = PredictionTask(UpGen(), AlwaysUp(), total_movements=3)
    task.play_game()

    assert len(task.bet_log) == 3, "bet_log should have one entry per step"

    for idx, (step, value, bet, received_reward) in enumerate(task.bet_log, start=1):
        assert step == idx, "Step indices should start at 1 and increase by 1"
        assert isinstance(value, (int, float)), "Logged value must be numeric"
        assert bet in {0, 1}, "Bet must be 0 or 1"
        assert received_reward in {-1, 1}, "Reward must be +1 or -1"


# ------------------------------------------------------------------------
# SECTION 5 — PredictionTask: Edge Cases
# ------------------------------------------------------------------------

def test_prediction_task_zero_steps_no_logs_and_zero_reward():
    """
    With total_movements=0:
    - play_game should return 0 (no rewards accumulated)
    - reward_development and bet_log should be empty
    """
    class AnyGen:
        def generate_value(self, last): return last + 1

    class AnyAgent:
        def observe(self, v): pass
        def place_bet(self): return 1

    task = PredictionTask(AnyGen(), AnyAgent(), total_movements=0)
    reward = task.play_game()
    assert reward == 0, "No steps → zero cumulative reward"
    assert len(task.reward_development) == 0, "No steps → no reward development entries"
    assert len(task.bet_log) == 0, "No steps → no bet log entries"


def test_prediction_task_ties_are_penalized():
    """
    If the generator returns the SAME value as last_value (a 'tie'),
    the current implementation penalizes the agent (received_reward = -1).
    This test verifies that behavior explicitly.
    """
    class FlatGen:
        def generate_value(self, last): return last  # no change

    class AlwaysUp:
        def observe(self, v): pass
        def place_bet(self): return 1

    task = PredictionTask(FlatGen(), AlwaysUp(), total_movements=3, start_value=10)
    reward = task.play_game()
    assert reward == -3, "Ties should penalize the agent under current rules"


def test_prediction_task_respects_custom_start_value():
    """
    Custom start_value should affect the first comparison.
    Using IncrementGen with start_value=10 produces sequence 11,12,...
    """
    class IncrementGen:
        def generate_value(self, last): return last + 1

    class AlwaysUp:
        def observe(self, v): pass
        def place_bet(self): return 1

    task = PredictionTask(IncrementGen(), AlwaysUp(), total_movements=2, start_value=10)
    reward = task.play_game()
    assert reward == 2, "With strictly increasing series, reward should be +2"


# ------------------------------------------------------------------------
# SECTION 6 — Deprecation Details (Message & Kwargs Forwarding)
# ------------------------------------------------------------------------

def test_betting_game_deprecation_message_and_forwarding():
    """
    - Ensure the deprecation warning message mentions 'PredictionTask'.
    - Ensure kwargs like start_value are forwarded to the new class.
    """
    class IncGen:
        def generate_value(self, last): return last + 1

    class UpAgent:
        def observe(self, v): pass
        def place_bet(self): return 1

    with pytest.warns(DeprecationWarning) as rec:
        game = BettingGame(IncGen(), UpAgent(), total_movements=1, start_value=41)
        result = game.play_game()

    # Check message contains new name
    messages = [str(w.message) for w in rec]
    assert any("PredictionTask" in m for m in messages), "Warning should reference PredictionTask"

    # Kwargs forwarding: start_value=41 → first generated = 42 → up bet → +1 reward
    assert result == 1, "start_value should be forwarded correctly via the wrapper"


    
