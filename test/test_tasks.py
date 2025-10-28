import pytest
import numpy as np
from tsdm.tasks import *


# ------------------------------------------------------------------------
# SECTION 1 — Base Class Integrity: Must Not Instantiate Directly
# ------------------------------------------------------------------------

def test_task_cannot_be_instantiated_directly():
    """
    Task is an abstract base class and should not be instantiable.
    Attempting to instantiate should raise a TypeError.
    """
    with pytest.raises(TypeError):
        Task(generator=None, agent=None, total_movements=5)


# ------------------------------------------------------------------------
# SECTION 2 — Must Implement Abstract Methods
# ------------------------------------------------------------------------

def test_task_requires_play_turn_and_play_game():
    """
    A subclass that does not implement play_turn or play_game must fail.
    """

    class IncompleteTask(Task):
        pass  # does not override abstract methods

    with pytest.raises(TypeError):
        IncompleteTask(generator=None, agent=None, total_movements=5)


def test_task_subclass_with_methods_can_instantiate():
    """
    A subclass that *does* implement play_turn and play_game should construct.
    """

    class DummyTask(Task):
        def play_turn(self, step):
            pass
        def play_game(self):
            return 0

    task = DummyTask(generator=None, agent=None, total_movements=3)
    assert isinstance(task, DummyTask)


# ------------------------------------------------------------------------
# SECTION 3 — Base Attributes and Log Initialization
# ------------------------------------------------------------------------

def test_task_initial_attributes_and_log_start_empty():
    """
    Task should initialize:
      - generator stored
      - agent stored
      - total_movements stored
      - log starts empty list
    """

    class DummyTask(Task):
        def play_turn(self, step): pass
        def play_game(self): return 0

    g = object()
    a = object()
    task = DummyTask(generator=g, agent=a, total_movements=10)

    assert task.generator is g
    assert task.agent is a
    assert task.total_movements == 10
    assert isinstance(task.log, list)
    assert len(task.log) == 0, "Log must be empty at initialization"


# ------------------------------------------------------------------------
# SECTION 4 — Deprecation: BettingGame → PredictionTask
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


# ------------------------------------------------------------------------
# SECTION 5 — PredictionTask: Core Reward Behavior (Parametrized)
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



# ------------------------------------------------------------------------
# SECTION 6 — PredictionTask: Observation & State Flow
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
# SECTION 7 — PredictionTask: Logging & Development Tracking
# ------------------------------------------------------------------------

def test_prediction_task_reward_development_length_and_values():
    """
    reward_development should:
    - have one entry per step
    - reflect cumulative reward correctly
    Using a strictly increasing generator + always-up agent → [1, 2, 3, 4]
    """
    class UpGenerator:
        def __init__(self): self.v = 0
        def generate_value(self, last):
            self.v += 1
            return self.v

    class AlwaysUp:
        def observe(self, v): pass
        def place_bet(self): return 1

    task = PredictionTask(UpGenerator(), AlwaysUp(), total_movements=4, start_value=0)
    task.play_game()
    assert len(task.reward_development) == 4, "Must log one cumulative entry per step"
    assert list(task.reward_development) == [1, 2, 3, 4], "Cumulative reward should increase by +1 each step"


def test_prediction_task_log_structure_and_step_indices():
    """
    task.log entries: dicts with keys
      - t (int)
      - value_prev (float)
      - value (float)
      - bet (0 or 1)
      - received_reward (+1 or -1)
      - reward_cum (int cumulative)
    
    Checks:
    - log length == total steps
    - step indices start at 1 and increase by 1
    - logged values are numeric
    - bet is 0/1
    - reward is ±1
    - cumulative reward is consistent
    """

    class UpGenerator:
        def __init__(self): self.v = 0
        def generate_value(self, last):
            self.v += 1
            return self.v

    class AlwaysUp:
        def observe(self, _): pass
        def place_bet(self): return 1

    task = PredictionTask(UpGenerator(), AlwaysUp(), total_movements=3)
    final_reward = task.play_game()

    # 1) correct number of records
    assert len(task.log) == 3, "log should have one entry per step"

    # 2) check fields and values
    for idx, rec in enumerate(task.log, start=1):
        assert rec["t"] == idx, "Step index t should start at 1 and increment by 1"
        assert isinstance(rec["value_prev"], (int, float)), "value_prev must be numeric"
        assert isinstance(rec["value"], (int, float)), "value must be numeric"
        assert rec["bet"] in {0, 1}, "bet must be 0 or 1"
        assert rec["received_reward"] in {-1, 1}, "received_reward must be ±1"
        # Cumulative reward must match direct cumulative sum
        assert rec["reward_cum"] == sum(r["received_reward"] for r in task.log[:idx])

    # 3) final cumulative reward matches environment’s returned result
    assert final_reward == task.log[-1]["reward_cum"], "Final returned reward must match last cumulative reward"


def test_prediction_task_bet_log_deprecated_warning():
    """
    accessing bet_log should:
    - emit DeprecationWarning
    - return tuples (step, value, bet, received_reward)
    """

    class UpGenerator:
        def __init__(self): self.v = 0
        def generate_value(self, last):
            self.v += 1
            return self.v

    class AlwaysUp:
        def observe(self, _): pass
        def place_bet(self): return 1

    task = PredictionTask(UpGenerator(), AlwaysUp(), total_movements=3)
    task.play_game()

    with pytest.warns(DeprecationWarning):
        bet_log = task.bet_log

    assert len(bet_log) == 3, "bet_log should reflect number of steps"
    for (step, value, bet, received_reward) in bet_log:
        assert bet in {0, 1}
        assert received_reward in {-1, 1}


# ------------------------------------------------------------------------
# SECTION 8 — PredictionTask: Edge Cases
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
    assert len(task.log) == 0, "No steps → no bet log entries"


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
# SECTION 9 — AllocationTask: Budget Dynamics (Parametrized)
# ------------------------------------------------------------------------


# --- Dummy Allocation Generators -------------------------------

class UpGen:
    def generate_value(self, last):
        return (0.0 if last is None else last) + 1.0

class DownGen:
    def generate_value(self, last):
        return (0.0 if last is None else last) - 1.0


# --- Dummy Allocation Agents (vector outputs) ---------------------------

class EqualWeightAgent:
    """Always returns equal allocation across two generators."""
    def observe(self, _): pass
    def place_bet(self): return np.array([0.5, 0.5], dtype=float)

class FirstOnlyAgent:
    """Allocates 100% to the first generator."""
    def observe(self, _): pass
    def place_bet(self): return np.array([1.0, 0.0], dtype=float)

class SecondOnlyAgent:
    """Allocates 100% to the second generator."""
    def observe(self, _): pass
    def place_bet(self): return np.array([0.0, 1.0], dtype=float)


@pytest.mark.parametrize(
    "generators, agent, steps, expected_relation",
    [
        # Both streams rising → any long-only allocation should grow budget (> 1)
        ([UpGen(), UpGen()],     FirstOnlyAgent(),   5, ">"),
        ([UpGen(), UpGen()],     SecondOnlyAgent(),  5, ">"),
        ([UpGen(), UpGen()],     EqualWeightAgent(), 5, ">"),

        # Both streams falling → any long-only allocation should shrink budget (< 1)
        ([DownGen(), DownGen()], FirstOnlyAgent(),   5, "<"),
        ([DownGen(), DownGen()], SecondOnlyAgent(),  5, "<"),
        ([DownGen(), DownGen()], EqualWeightAgent(), 5, "<"),

        # Mixed regime → outcome depends on which stream you allocate to
        ([UpGen(), DownGen()],   FirstOnlyAgent(),   5, ">"),
        ([UpGen(), DownGen()],   SecondOnlyAgent(),  5, "<"),
        ([UpGen(), DownGen()],   EqualWeightAgent(), 5, None),  # ambiguous sign; just assert != 1
    ]
)
def test_allocation_task_budget_sign(generators, agent, steps, expected_relation):
    """
    Parametrized correctness for AllocationTask:
    - With both streams rising, final budget should be > 1.0 (initial_budget).
    - With both falling, final budget should be < 1.0.
    - With mixed streams, sign depends on which stream is favored.
    """
    # Start values at zero for both streams (first step change is defined)
    start_values = np.array([10.0, 10.0], dtype=float)

    task = AllocationTask(
        generators=generators,
        agent=agent,
        total_movements=steps,
        start_values=start_values,
        initial_budget=1.0,
        tc=0.0,
    )
    final_budget = task.play_game()

    if expected_relation == ">":
        assert final_budget > 1.0, f"Expected budget to grow (>1), got {final_budget}"
    elif expected_relation == "<":
        assert final_budget < 1.0, f"Expected budget to shrink (<1), got {final_budget}"
    else:
        # Ambiguous case: should differ from initial if any movement occurred
        assert not np.isclose(final_budget, 1.0), "Budget should change under mixed streams with reallocation"


# ------------------------------------------------------------------------
# SECTION 10 — AllocationTask: Observation, Logging, and Constraints
# ------------------------------------------------------------------------

# --- Simple generators that respect `last` ---

class FlatGen:
    def generate_value(self, last):
        return (0.0 if last is None else last)

# --- Helper agents ---
class RecordingAgent:
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)
        self.observed = []
    def observe(self, v):
        # v is the previous values vector
        self.observed.append(np.array(v, dtype=float))
    def place_bet(self):
        return self.weights

class BadAgent:  # returns invalid weights to test projection
    def observe(self, v): pass
    def place_bet(self):
        return np.array([-0.2, 1.6])  # negative + sums > 1

class ToggleAgent:  # alternates allocations to induce turnover
    def __init__(self):
        self.k = 0
    def observe(self, v): pass
    def place_bet(self):
        self.k += 1
        return np.array([1.0, 0.0]) if self.k % 2 else np.array([0.0, 1.0])

def _alloc_task(generators, agent, steps, start_vals=(10.0, 10.0), tc=0.0, initial_budget=1.0):
    return AllocationTask(
        generators=generators,
        agent=agent,
        total_movements=steps,
        start_values=np.array(start_vals, dtype=float),
        initial_budget=initial_budget,
        tc=tc,
    )

def test_allocation_task_observe_receives_previous_values():
    """
    The agent must receive the *current* generator values via `observe(values)`
    *before* each allocation decision.

    For a 2-generator AllocationTask with UpGen() (value increases by +1 per step)
    and start_values = [5, 7], the value trajectory across steps is:

        Step 1: observe [5, 7]      → new values become [6, 8]
        Step 2: observe [6, 8]      → new values become [7, 9]
        Step 3: observe [7, 9]      → new values become [8, 10]

    The agent's `observe()` should therefore receive exactly the sequence:
        [ [5, 7], [6, 8], [7, 9] ].

    This confirms:
    - Observation happens before generator updates.
    - Values passed to the agent match the task's internal state.
    """
    gens = [UpGen(), UpGen()]
    start = np.array([5.0, 7.0])
    agent = RecordingAgent(weights=[0.5, 0.5])

    task = _alloc_task(gens, agent, steps=3, start_vals=start)
    task.play_game()

    expected = [
        start,
        start + 1.0,
        start + 2.0,
    ]

    assert len(agent.observed) == 3, "Agent must receive one observation per step"
    for got, exp in zip(agent.observed, expected):
        assert np.allclose(got, exp), f"Observed {got}, expected {exp}"



def test_allocation_task_log_schema_and_lengths():
    """
    AllocationTask log structure validation.

    For each step, AllocationTask records a dictionary into `task.log` containing:
        {
            "t": step index (1-based),
            "values": ndarray (n,),
            "relative_changes": ndarray (n,),
            "allocations": ndarray (n,),
            "turnover": float,
            "combined_change": float,
            "new_budget": float,
        }

    Additionally:
        - `task.log` should contain one entry per step.
        - `task.budget_development` stores the budget trajectory and therefore has
          length (steps + 1), because it includes the initial budget at index 0.
        - The cumulative budget from `task.log[-1]["new_budget"]` must match the
          final return value of `play_game()`.

    This test ensures:
        * Log schema stability (important for analysis / pandas workflows).
        * Budget trajectory consistency.
        * Simplex constraints on allocations (non-negative, sum to 1).
    """
    gens = [UpGen(), UpGen()]
    agent = RecordingAgent(weights=[1.0, 0.0])  # Always allocate entirely to first stream
    task = _alloc_task(gens, agent, steps=3)

    final_budget = task.play_game()

    # 1) Logging counts
    assert len(task.log) == 3, "Expected one log entry per step"
    assert len(task.budget_development) == 4, (
        "budget_development should include the initial budget + one per step"
    )

    # 2) Log schema + cumulative budget tracking
    for idx, rec in enumerate(task.log, start=1):
        # Check required keys exist
        for key in [
            "t", "values", "relative_changes", "allocations",
            "turnover", "combined_change", "new_budget"
        ]:
            assert key in rec, f"Missing '{key}' in allocation log record at step {idx}"

        # Step index matches
        assert rec["t"] == idx, "Step index must increment starting from 1"

        # Allocations must be a valid probability simplex vector
        alloc = rec["allocations"]
        assert alloc.ndim == 1, "allocations must be a 1D vector"
        assert np.all(alloc >= 0), "allocations must be non-negative"
        assert np.isclose(alloc.sum(), 1.0), "allocations must sum to 1.0"

        # Budget trajectory must align with log values
        assert np.isclose(task.budget_development[idx], rec["new_budget"]), (
            "budget_development must match new_budget progression"
        )

    # 3) Final budget agrees with last logged budget
    assert np.isclose(final_budget, task.log[-1]["new_budget"]), (
        "Final returned budget must equal last cumulative logged budget"
    )



def test_allocation_task_simplex_projection_enforced():
    """
    Allocation vectors returned by the agent must always be projected onto the
    probability simplex:

        allocations[i] >= 0  for all i
        sum(allocations) == 1

    This is important because agents are *not* required to output valid weights.
    They may output:
        - negative values
        - values that sum to > 1 or < 1
        - arbitrary real-valued vectors

    AllocationTask must therefore enforce feasibility by projecting the agent's
    output to the simplex before computing returns.

    This test uses `BadAgent`, which intentionally outputs an invalid vector
    (e.g., negative values or values that don’t sum to 1). We confirm that the
    logged allocations have been corrected by the projection method.
    """
    gens = [UpGen(), UpGen()]
    task = _alloc_task(gens, BadAgent(), steps=1)

    task.play_game()

    # Retrieve allocations actually applied in the first (and only) logged step
    w = task.log[0]["allocations"]

    # Allocations must be valid after projection
    assert np.all(w >= 0.0), (
        "Allocations must be non-negative after simplex projection"
    )
    assert np.isclose(w.sum(), 1.0), (
        "Allocations must sum to 1.0 after simplex projection"
    )


def test_allocation_task_zero_steps_no_logs_and_budget_unchanged():
    """
    When total_movements = 0, the task should:
      - perform no generator updates,
      - never call agent.place_bet(),
      - never modify the budget,
      - produce *no* step logs,
      - but preserve the initial budget inside budget_development.

    This confirms that the task handles the degenerate case cleanly and does
    not introduce accidental state transitions or side effects.
    """
    gens = [UpGen(), UpGen()]
    agent = RecordingAgent([0.5, 0.5])

    # Explicit non-default initial_budget for clearer verification
    task = _alloc_task(gens, agent, steps=0, initial_budget=1.23)

    final_budget = task.play_game()

    # Budget must remain exactly unchanged
    assert np.isclose(final_budget, 1.23), (
        "With zero steps, the final budget must equal the initial budget."
    )

    # No per-step logs should be produced
    assert len(task.log) == 0, (
        "No steps executed → log must remain empty."
    )

    # budget_development should contain only one entry (the initial state)
    assert len(task.budget_development) == 1, (
        "budget_development should contain only the initial budget when no steps are run."
    )
    assert np.isclose(task.budget_development[0], 1.23), (
        "budget_development must preserve the initial budget exactly."
    )



def test_allocation_task_turnover_cost_penalizes_rebalancing():
    """
    Turnover cost (`tc`) should reduce final budget when the agent changes
    allocations over time.

    We use `ToggleAgent`, which alternates:
        [1, 0] → [0, 1] → [1, 0] → ...
    This produces maximum allocation turnover each step.

    We run two AllocationTasks:
      1) With tc = 0.05   (turnover is penalized)
      2) With tc = 0.0    (no penalty)

    Since turnover strictly increases the cost term:

        new_budget = budget * (1 + combined_change - tc * turnover)

    The final budget with tc > 0 must be *less than or equal to* the budget with tc = 0,
    all else equal.
    """
    gens = [UpGen(), UpGen()]
    steps = 6
    start = np.array([10.0, 10.0], dtype=float)

    # High turnover version
    task_hi_tc = _alloc_task(
        generators=gens,
        agent=ToggleAgent(),
        steps=steps,
        start_vals=start,
        tc=0.05,
    )
    b_hi = task_hi_tc.play_game()

    # Same environment + agent behavior, but without transaction costs
    task_no_tc = _alloc_task(
        generators=gens,
        agent=ToggleAgent(),
        steps=steps,
        start_vals=start,
        tc=0.0,
    )
    b_no = task_no_tc.play_game()

    # Turnover cost must *never* increase wealth
    assert b_hi <= b_no, (
        "Turnover penalty must reduce (or at most leave unchanged) the final budget. "
        f"Got: no-cost={b_no:.6f}, with-cost={b_hi:.6f}"
    )



def test_allocation_task_monotonic_budget_when_up_and_constant_weights_no_tc():
    """
    If:
      - All generator streams are monotonically increasing, and
      - The agent uses *fixed* allocations (no turnover), and
      - Transaction cost tc = 0,

    then the budget must be **monotonically non-decreasing** over time.

    Reason:
      Relative changes are positive each step:
          values_t > values_{t-1}  →  relative_change > 0

      The agent does not rebalance, so turnover = 0 each step.

      Budget update rule becomes:
          new_budget = budget * (1 + combined_change)

      with combined_change >= 0 → new_budget >= budget.

    This test ensures that:
      - Positive trends increase (or maintain) wealth
      - Zero transaction cost does not introduce artificial drag
      - The budget_development trace correctly logs each step change
    """
    gens = [UpGen(), UpGen()]  # strictly increasing streams
    agent = RecordingAgent([0.7, 0.3])  # constant allocations, no turnover
    task = _alloc_task(gens, agent, steps=5, tc=0.0)

    task.play_game()
    dev = task.budget_development  # length = steps + 1 (includes initial budget)

    # Check non-decreasing behavior:
    assert all(dev[i+1] >= dev[i] for i in range(len(dev) - 1)), (
        "Budget should be non-decreasing when all series rise and tc=0, "
        f"but got development: {dev}"
    )



def test_allocation_task_invalid_start_values_length_raises():
    """
    start_values must match the number of generators exactly.

    AllocationTask initializes `values` and `last_values` from start_values.
    If the provided start_values array does not have length == number of generators,
    the task cannot determine which value corresponds to which generator, and must
    raise a ValueError to prevent silent misalignment.

    This test ensures correctness of input validation and protects against
    subtle bugs caused by mismatched dimensions.
    """
    gens = [UpGen(), UpGen(), UpGen()]  # 3 generators
    agent = RecordingAgent([1.0, 0.0, 0.0])

    # start_values must also have length 3 — here we intentionally violate it
    with pytest.raises(ValueError):
        AllocationTask(
            generators=gens,
            agent=agent,
            total_movements=1,
            start_values=np.array([10.0, 10.0]),  # wrong length → must raise
        )



def test_allocation_task_requires_two_generators_raises():
    """
    AllocationTask is defined for *multi*-generator allocation problems.
    At least two generators are required to form a meaningful allocation vector.

    If only one generator is provided:
      - No allocation choice exists (the simplex would collapse to [1.0]),
      - Turnover and diversification effects are undefined,
      - The task reduces to a trivial scalar growth problem and should instead
        be modeled using PredictionTask or a simpler single-stream setup.

    Therefore, the constructor must raise a ValueError when fewer than two
    generators are supplied. This test ensures the correct early failure behavior.
    """
    agent = RecordingAgent([1.0])  # Shape/sum irrelevant; constructor should fail before use.

    with pytest.raises(ValueError):
        AllocationTask(
            generators=[UpGen()],   #  only one generator — not allowed
            agent=agent,
            total_movements=1,
            start_values=np.array([10.0]),
        )



def test_allocation_task_projection_edge_all_zero_vector_becomes_equal_weights():
    """
    Allocation vectors must be valid probability distributions:
        - entries >= 0
        - sum = 1

    In practice, an agent may return an invalid vector — including the extreme case
    where all proposed weights are zero (e.g., uninitialized policy, numerical collapse,
    exploration errors, etc.).

    The AllocationTask enforces feasibility via `_project_to_simplex`, which ensures:
        - If the raw vector is all zeros, it is projected to *uniform equal weights*.
          (This is the maximum-entropy neutral allocation on the simplex.)

    This test verifies that behavior explicitly.
    """
    class ZeroAgent:
        def observe(self, v): pass
        def place_bet(self): return np.array([0.0, 0.0])  # invalid — sum = 0

    gens = [UpGen(), UpGen()]
    task = _alloc_task(gens, ZeroAgent(), steps=1)
    task.play_game()

    w = task.log[0]["allocations"]
    assert np.allclose(w, np.array([0.5, 0.5])), \
        "All-zero proposed allocation should project to equal weights on the simplex"


    
# ------------------------------------------------------------------------
# SECTION 11 — ExecutionTask: Helpers (Generators, Agents, Factory)
# ------------------------------------------------------------------------

# We reuse UpGen and FlatGen from above where helpful:
# - UpGen: values increase by +1 each step (per index)
# - FlatGen: values stay the same (no change across time)

class ConstGen:
    """Generator that always returns a constant scalar value (ignores `last`)."""
    def __init__(self, value: float):
        self.value = float(value)
    def generate_value(self, last):
        return self.value


class FractionExecutorAgent:
    """
    Executes a fixed fraction of the remaining scalar resource each step
    but split evenly across K generators (for simplicity).
    For K=1, it just returns [fraction * remaining].
    """
    def __init__(self, fraction: float, k: int):
        self.fraction = float(fraction)
        self.k = int(k)
        self.observed = []
    def observe(self, state):
        # state is a dict with keys: values, remaining_inventory, remaining_time, cost_cum
        self.observed.append(state)
    def place_bet(self):
        remain = self.observed[-1]["remaining_inventory"]
        per = (self.fraction * remain) / self.k if self.k > 0 else 0.0
        return np.full(self.k, per, dtype=float)


class AllAtOnceAgent:
    """Consumes the entire remaining scalar resource immediately, split evenly across K."""
    def __init__(self, k: int):
        self.k = int(k)
    def observe(self, state): pass
    def place_bet(self):
        # Remaining is read from last observed state:
        # This agent assumes the task provides it; if not, it's a no-op.
        # We therefore implement a safe default via a closure pattern in tests.
        raise RuntimeError("AllAtOnceAgent requires a wrapper to inject remaining.")


class OverConsumeAgent:
    """Proposes absurdly large nonnegative quantities (to test projection to remaining)."""
    def __init__(self, k: int, scale: float = 10.0):
        self.k = int(k)
        self.scale = float(scale)
    def observe(self, state): self._last_state = state
    def place_bet(self):
        # Propose a vector much larger than remaining to force projection
        remain = self._last_state["remaining_inventory"]
        return np.full(self.k, self.scale * remain, dtype=float)


def _exec_task(
    generators,
    agent,
    steps,
    start_vals,
    initial_inventory=1.0,
):
    """
    Helper to construct an ExecutionTask with common defaults.
    - generators: list of K generators
    - agent: execution agent
    - steps: total_movements (horizon)
    - start_vals: np.array of length K
    - initial_inventory: initial scalar inventory/resource

    """
    return ExecutionTask(
        generators=generators,
        agent=agent,
        total_movements=steps,
        start_values=np.array(start_vals, dtype=float),
        initial_inventory=float(initial_inventory)
        )


# ------------------------------------------------------------------------
# SECTION 12 — ExecutionTask: Observation & State
# ------------------------------------------------------------------------

def test_execution_task_observe_includes_remaining_time_and_inventory():
    """
    The agent must observe the state BEFORE acting each step, including:
      - values (vector across K generators)
      - remaining_inventory (scalar)
      - remaining_time (steps left including this one)
      - cost_cum (cumulative cost so far)

    We verify the presence and evolution of these fields across steps.
    """
    K = 2
    gens = [UpGen(), UpGen()]                    # each value increments by +1 per step
    start = np.array([10.0, 20.0], dtype=float)  # t=0 values
    agent = FractionExecutorAgent(fraction=0.2, k=K)
    task = _exec_task(gens, agent, steps=3, start_vals=start, initial_inventory=1.0)

    task.play_game()

    # 3 steps → 3 observations
    assert len(agent.observed) == 3, "Agent must receive one observation per step"

    # remaining_time should count down: 3, 2, 1
    rem_times = [s["remaining_time"] for s in agent.observed]
    assert rem_times == [3, 2, 1], f"remaining_time should decrement; got {rem_times}"

    # values should be observed BEFORE update each step:
    # Step 1 observes start
    # Step 2 observes start + 1
    # Step 3 observes start + 2
    expected_values = [start, start + 1.0, start + 2.0]
    for got, exp in zip([s["values"] for s in agent.observed], expected_values):
        assert np.allclose(got, exp), f"Observed values {got} != expected {exp}"

    # remaining_inventory is positive and non-increasing
    rems = [s["remaining_inventory"] for s in agent.observed]
    assert all(r >= 0 for r in rems), "remaining_inventory must be non-negative"
    assert all(rems[i+1] <= rems[i] for i in range(len(rems)-1)), \
        f"remaining_inventory must not increase; got {rems}"


# ------------------------------------------------------------------------
# SECTION 13 — ExecutionTask: Core Mechanics
# ------------------------------------------------------------------------

def test_execution_task_stops_when_fully_consumed():
    """
    If the agent consumes the entire remaining resource at the first step,
    the task should:
      - log exactly one step,
      - set remaining to 0,
      - stop early.
    """
    K = 2
    gens = [ConstGen(1.0), ConstGen(1.0)]
    start = np.array([1.0, 1.0])

    # Wrap AllAtOnceAgent to inject "remaining" into its output evenly across K.
    class AllAtOnceWrapper(AllAtOnceAgent):
        def __init__(self, k): super().__init__(k); self._remain = None
        def observe(self, state): self._remain = state["remaining_inventory"]
        def place_bet(self): return np.full(self.k, self._remain / self.k)

    agent = AllAtOnceWrapper(k=K)
    task = _exec_task(gens, agent, steps=10, start_vals=start, initial_inventory=5.0)

    reward = task.play_game()
    # cost is negative reward, but we only care about termination behavior here
    assert len(task.log) == 1, "Should execute only one step when fully consumed at once"
    assert task.log[0]["q_remaining"] == 0.0, "Remaining must be zero after full consumption"


def test_execution_task_projects_overconsumption_to_remaining():
    """
    The environment must cap total proposed consumption to the remaining resource.
    OverConsumeAgent proposes a vector much larger than remaining. The task should:
      - scale it down so sum(x) == previous remaining
      - set new remaining to 0
    """
    K = 3
    gens = [ConstGen(0.0) for _ in range(K)]  # zero value ⇒ price_diff cost term = 0
    start = np.zeros(K, dtype=float)
    agent = OverConsumeAgent(k=K, scale=10.0)
    task = _exec_task(gens, agent, steps=1, start_vals=start, initial_inventory=2.5)

    task.play_game()
    x = task.log[0]["x"]
    assert np.isclose(x.sum(), 2.5), "Total executed must equal previous remaining when overproposed"
    assert task.log[0]["q_remaining"] == 0.0, "Remaining must be zero after capping to remaining"


def test_execution_task_cost_uses_next_values():
    """
    With the updated execution logic, the step cost is computed as:
        step_cost = dot(values_{t+1} (post-advance), x_t)

    Since steps=1, this is the FINAL step, so any remaining inventory is
    liquidated evenly regardless of the agent's suggestion.

    Setup:
      - K = 2 generators, both UpGen → each increases value by +1
      - start values = [10.0, 20.0]
      - next values = [11.0, 21.0]
      - initial inventory Q = 1.0
      - Final step → full liquidation evenly → x = [0.5, 0.5]

    Expected step cost:
      cost = 11.0 * 0.5 + 21.0 * 0.5 = 16.0


    """
    K = 2
    gens = [UpGen(), UpGen()]
    start = np.array([10.0, 20.0], dtype=float)

    agent = FractionExecutorAgent(fraction=0.4, k=K)

    task0 = _exec_task(gens, agent, steps=1, start_vals=start, initial_inventory=1.0)
    reward0 = task0.play_game()
    cost0 = -reward0
    assert np.isclose(cost0, 16.0), f"Expected execution cost 16.0, got {cost0}"


# ------------------------------------------------------------------------
# SECTION 14 — ExecutionTask: Logging Schema & Consistency
# ------------------------------------------------------------------------

def test_execution_task_log_schema_and_consistency():
    """
    ExecutionTask must log one dictionary per step with keys:
        - t: 1-based step index
        - values_prev: ndarray (K,)
        - values: ndarray (K,)
        - x: ndarray (K,)   [executed non-negative vector]
        - q_remaining: float
        - step_cost: float
        - cost_cum: float

    We check:
      * Correct key presence
      * Non-negativity of x
      * q_remaining decreases by sum(x)
      * cost_cum is cumulative sum of step_cost
      * Number of logs equals steps taken
    """
    K = 2
    gens = [UpGen(), UpGen()]
    start = np.array([1.0, 1.0], dtype=float)
    agent = FractionExecutorAgent(fraction=0.5, k=K)

    task = _exec_task(gens, agent, steps=3, start_vals=start, initial_inventory=1.0)
    reward = task.play_game()

    # logs exist (might be fewer than steps if fully consumed early)
    assert len(task.log) >= 1, "Must log at least the first step"
    keys_required = {"t", "values_prev", "values", "x", "q_remaining", "step_cost", "cost_cum"}

    cost_sum = 0.0
    q_prev = 1.0
    for idx, rec in enumerate(task.log, start=1):
        assert keys_required.issubset(rec.keys()), f"Missing keys in log record: {rec.keys()}"
        x = rec["x"]
        assert np.all(x >= 0.0), "Executed vector x must be non-negative"
        # q_remaining decreases by sum(x) relative to the previous remaining
        q_after = rec["q_remaining"]
        assert np.isclose(q_after, q_prev - x.sum()), "Remaining must decrease by sum(x)"
        q_prev = q_after

        # cumulative cost matches sum of step_costs
        cost_sum += rec["step_cost"]
        assert np.isclose(rec["cost_cum"], cost_sum), "cost_cum must be cumulative sum of step_cost"

        # sanity on indices
        assert rec["t"] == idx, "t should be 1-based and incrementing"

    # reward = - total cost
    assert np.isclose(-reward, cost_sum), "Returned reward must equal negative cumulative cost"


# ------------------------------------------------------------------------
# SECTION 15 — ExecutionTask: Edge Cases
# ------------------------------------------------------------------------

def test_execution_task_zero_steps_no_logs_and_zero_cost():
    """
    With total_movements=0:
      - No observations, no actions, no logs.
      - Cost is zero; reward is zero.
    """
    K = 2
    gens = [FlatGen(), FlatGen()]
    start = np.array([5.0, 7.0], dtype=float)
    agent = FractionExecutorAgent(fraction=0.5, k=K)
    task = _exec_task(gens, agent, steps=0, start_vals=start, initial_inventory=3.0)

    reward = task.play_game()
    assert reward == 0.0, "Zero steps should yield zero cost (zero reward)"
    assert len(task.log) == 0, "No steps ⇒ no logs"


def test_execution_task_cost_with_flat_values_and_forced_liquidation():
    """
    With flat values (no change) and no running penalty:
      - Step cost equals the executed cash that step: values · x.
      - Using K=1, value=10.0 constant, Q=4.0, f=0.5, steps=2.
      - The agent proposes:
          * step 1: x = f * Q = 2.0
          * step 2: x = f * (Q - 2.0) = 1.0 (proposal)
        BUT on the final step the task forces liquidation of any remaining inventory,
        so step 2 executes x = 2.0 (the remaining quantity), not 1.0.
      - Hence costs:
          * step 1: 10 * 2.0 = 20.0
          * step 2: 10 * 2.0 = 20.0
        Total = 40.0
    """
    K = 1
    Q = 4.0
    f = 0.5

    gens = [FlatGen()]
    start = np.array([10.0], dtype=float)

    agent = FractionExecutorAgent(fraction=f, k=K)
    task = _exec_task(gens, agent, steps=2, start_vals=start, initial_inventory=Q)
    reward = task.play_game()
    cost = -reward  # task returns negative cost as reward

    # Totals
    assert np.isclose(cost, 40.0), f"Execution cost mismatch: got {cost}, expected 40.0"

    # Log sanity checks
    assert len(task.log) == 2, "Should have exactly two step logs"

    # Step 1: execute 2.0 at price 10.0 → cost 20.0, remaining 2.0
    rec1 = task.log[0]
    assert np.isclose(rec1["x"].sum(), 2.0), "Step 1 executed quantity should be 2.0"
    assert np.isclose(rec1["step_cost"], 20.0), "Step 1 cost should be 20.0"
    assert np.isclose(rec1["q_remaining"], 2.0), "Remaining after step 1 should be 2.0"
    assert np.isclose(rec1["cost_cum"], 20.0), "Cumulative cost after step 1 should be 20.0"

    # Step 2 (final): forced liquidation executes the remaining 2.0 at price 10.0 → cost 20.0, remaining 0.0
    rec2 = task.log[1]
    assert np.isclose(rec2["x"].sum(), 2.0), "Final step must liquidate remaining 2.0"
    assert np.isclose(rec2["step_cost"], 20.0), "Step 2 cost should be 20.0"
    assert np.isclose(rec2["q_remaining"], 0.0), "Remaining after final step should be 0.0"
    assert np.isclose(rec2["cost_cum"], 40.0), "Cumulative cost after final step should be 40.0"
