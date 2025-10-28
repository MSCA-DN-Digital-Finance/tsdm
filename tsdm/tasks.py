from warnings import warn
from abc import ABC, abstractmethod
from typing import Any, Sequence, List, Dict, Optional
import numpy as np

# ========================================================================
# Abstract Base Class for Tasks
# ========================================================================


class Task(ABC):
    """
    Abstract base class for time-series decision-making tasks.

    Subclasses MUST implement:
      - play_turn(step): execute one step/turn of the task.
      - play_game():     run the full task and return a final scalar metric.
    """

    def __init__(self, generator: Any, agent: Any, total_movements: int):
        self.generator = generator
        self.agent = agent
        self.total_movements = int(total_movements)
        self.log: List[Dict[str, Any]] = []

    @abstractmethod
    def play_turn(self, step: int) -> None:
        """Execute one turn/step of the task."""
        raise NotImplementedError

    @abstractmethod
    def play_game(self) -> Any:
        """Run the full task and return a final scalar metric (e.g., reward/wealth)."""
        raise NotImplementedError


# ========================================================================
# Prediction Task
# ========================================================================


class PredictionTask(Task):
    """
    Binary directional prediction task (domain-agnostic logging).

    Generator contract:
        value_t = generator.generate_value(value_{t-1})

    Agent contract (same pattern as AllocationTask):
        agent.observe(last_value)          # called before each turn
        bet = agent.place_bet()            # 1 = up, 0 = down

    Logging (one dict per step in `self.log`):
        {
            "t": int,                      # step index (1-based)
            "value_prev": float,           # previous value
            "value": float,                # new value
            "bet": int,                    # 0 or 1
            "received_reward": int,        # +1 or -1
            "reward_cum": int,             # cumulative reward after this step
        }
    """

    def __init__(self, generator, agent, total_movements, start_value=0):
        super().__init__(generator=generator, agent=agent, total_movements=total_movements)
        self.last_value = start_value
        self.reward = 0
        self.reward_development = np.array([], dtype=int)  # cumulative reward per step
        

    def play_turn(self, step: int) -> None:
        """
        One turn:
        - Generate next value from generator.
        - Get agent's bet.
        - Compute reward (+1/-1).
        - Log using dict schema compatible with AllocationTask's approach.
        """
        value = self.generator.generate_value(self.last_value)
        bet = int(self.agent.place_bet())

        received_reward = 1 if (
            (value > self.last_value and bet == 1) or
            (value < self.last_value and bet == 0)
        ) else -1

        self.reward += received_reward
        self.reward_development = np.append(self.reward_development, self.reward)

        # dict-style log (self-describing, DataFrame-friendly)
        self.log.append({
            "t": step,
            "value_prev": float(self.last_value),
            "value": float(value),
            "bet": bet,
            "received_reward": int(received_reward),
            "reward_cum": int(self.reward),
        })

        self.last_value = value

    def play_game(self) -> int:
        """
        Run the full task.
        Agent observes the previous value before each turn.
        Returns final cumulative reward (int).
        """
        for step in range(1, self.total_movements + 1):
            if hasattr(self.agent, "observe"):
                self.agent.observe(self.last_value)
            self.play_turn(step)
        return self.reward

    # ---- Back-compat shim: old tuple-style bet_log -----------------------
    @property
    def bet_log(self):
        """
        DEPRECATED: use `self.log` instead.
        Returns a tuple view: (step, value, bet, received_reward).
        """
        warn(
            "PredictionTask.bet_log is deprecated; use `PredictionTask.log` "
            "with dict records instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return [
            (rec["t"], rec["value"], rec["bet"], rec["received_reward"])
            for rec in self.log
        ]


class BettingGame(PredictionTask):
    """
    DEPRECATED: Use `PredictionTask` instead. Will be removed in version 0.3.0.
    """

    def __init__(self, *args, **kwargs):
        warn(
            "BettingGame is deprecated and will be removed in v0.3.0. "
            "Use PredictionTask instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)



# ========================================================================
# Allocation Task
# ========================================================================


class AllocationTask(Task):
    """
    Multi-generator allocation task (domain-agnostic, long-only).

    Interface (same pattern as PredictionTask):
      - Before each step:
            agent.observe(values)
      - Then AllocationTask calls:
            allocations = agent.place_bet()
        where `allocations` is a 1D vector of length n_generators
        with allocations[i] >= 0 and sum(allocations) == 1.

    Mechanics:
      - You pass 2+ generators. Each exposes:
            gen.generate_value(last_value_i) -> next_value_i
      - We compute elementwise relative changes:
            relative_changes = (values_t / values_{t-1}) - 1
        (0 for the very first step if no previous values)
      - Budget update with turnover penalty:
            combined_change = allocations · relative_changes
            turnover = L1(allocations - prev_allocations)
            new_budget = budget * max(0, 1 + combined_change - tc * turnover)

    Logs per step (in self.log):
        {
            "t": step,
            "values": np.ndarray shape (n,),
            "relative_changes": np.ndarray shape (n,),
            "allocations": np.ndarray shape (n,),
            "turnover": float,
            "combined_change": float,
            "new_budget": float,
        }
    """

    def __init__(
        self,
        generators: Sequence[Any],
        agent: Any,
        total_movements: int,
        start_values: Optional[Sequence[float]] = None,
        initial_budget: float = 1.0,
        tc: float = 0.0,  # proportional “transaction” cost on allocation turnover (L1)
    ):
        if len(generators) < 2:
            raise ValueError("AllocationTask requires at least two generators.")
        super().__init__(generator=generators, agent=agent, total_movements=total_movements)

        self.generators: List[Any] = list(generators)
        self.n: int = len(self.generators)

        # current and previous generator values
        if start_values is not None:
            if len(start_values) != self.n:
                raise ValueError("start_values length must match number of generators.")
            self.values = np.asarray(start_values, dtype=float)
            self.last_values = self.values.copy()
        else:
            self.values = np.zeros(self.n, dtype=float)
            self.last_values = None  # so first step uses 0 relative change

        # allocations and budget
        self.allocations = np.full(self.n, 1.0 / self.n)   # start equally allocated
        self.last_allocations = self.allocations.copy()
        self.budget = float(initial_budget)
        self.tc = float(tc)

        # logging
        self.budget_development: List[float] = [self.budget]
        self.t = 0

    # ---------- utilities ----------
    @staticmethod
    def _relative_change(curr: np.ndarray, prev: np.ndarray) -> np.ndarray:
        prev_safe = np.where(prev == 0.0, np.nan, prev)
        rc = curr / prev_safe - 1.0
        return np.where(np.isnan(rc), 0.0, rc)

    def _project_to_simplex(self, a: np.ndarray) -> np.ndarray:
        """
        Project to the probability simplex: a_i >= 0, sum(a) = 1.
        """
        a = np.asarray(a, dtype=float).reshape(self.n)
        if np.all(a >= 0) and np.isclose(a.sum(), 1.0):
            return a
        # standard O(n log n) projection
        u = np.sort(a)[::-1]
        cssv = np.cumsum(u)
        rho_idx = np.nonzero(u * (np.arange(1, self.n + 1)) > (cssv - 1.0))[0]
        if len(rho_idx) == 0:
            return np.full(self.n, 1.0 / self.n)
        rho = rho_idx[-1]
        theta = (cssv[rho] - 1.0) / (rho + 1.0)
        a_proj = np.maximum(a - theta, 0.0)
        s = a_proj.sum()
        return a_proj / s if s > 0 else np.full(self.n, 1.0 / self.n)

    # ---------- one step ----------
    def play_turn(self, step: int) -> None:
        # 0) Agent observes current values (same as PredictionTask pattern)
        if hasattr(self.agent, "observe"):
            self.agent.observe(self.values.copy())

        # 1) Generators produce next values
        new_values = np.array(
            [gen.generate_value(self.values[i]) for i, gen in enumerate(self.generators)],
            dtype=float
        )

        # 2) Compute relative changes (0 on first step if no last_values)
        if self.last_values is None:
            relative_changes = np.zeros(self.n, dtype=float)
        else:
            relative_changes = self._relative_change(new_values, self.values)

        # 3) Agent returns allocation vector via place_bet()
        raw_alloc = self.agent.place_bet()
        allocations = self._project_to_simplex(np.asarray(raw_alloc, dtype=float))

        # 4) Update budget with turnover penalty
        combined_change = float(np.dot(allocations, relative_changes))
        turnover = float(np.sum(np.abs(allocations - self.allocations)))
        new_budget = self.budget * max(0.0, 1.0 + combined_change - self.tc * turnover)

        # 5) Log & update state
        self.log.append({
            "t": step,
            "values": new_values.copy(),
            "relative_changes": relative_changes.copy(),
            "allocations": allocations.copy(),
            "turnover": turnover,
            "combined_change": combined_change,
            "new_budget": new_budget,
        })

        self.budget = new_budget
        self.budget_development.append(self.budget)
        self.last_allocations = self.allocations.copy()
        self.allocations = allocations
        self.last_values = self.values.copy()
        self.values = new_values
        self.t += 1

    # ---------- full run ----------
    def play_game(self) -> float:
        for step in range(1, self.total_movements + 1):
            self.play_turn(step)
        return self.budget



# ========================================================================
# Execution Task
# ========================================================================

import numpy as np
from typing import Any, Sequence, List, Optional

class ExecutionTask(Task):
    """
    Execution Task.

    Goal
    ----
    Execute an initial inventory Q over T steps across K generators.
    The agent chooses absolute order sizes x_t >= 0 per generator each step.
    We minimize execution cost; `play_game()` returns NEGATIVE total cost (higher is better).
    Remaining inventory is forced to zero at the final step.

    Agent API
    ---------
    observe(state) where state is:
        {
          "values": np.ndarray (K,),          # current prices before acting
          "remaining_inventory": float,       # q_t
          "remaining_time": int,              # steps left including this one
          "cost_cum": float,                  # cumulative cost so far
        }
    place_bet() -> np.ndarray (K,) of non-negative order sizes (absolute units).

    Logging (per step) — appended to self.log:
        {
          "t": int,                           # 1-based external step index
          "values_prev": np.ndarray (K,),
          "values": np.ndarray (K,),          # prices used for execution this step
          "x": np.ndarray (K,),               # executed sizes this step
          "q_remaining": float,               # remaining after execution
          "step_cost": float,                 # cost this step
          "cost_cum": float,                  # cumulative cost so far
        }
    """

    def __init__(
        self,
        generators: Sequence[Any],            # K venues (each has .generate_value(last_value))
        agent: Any,
        total_movements: int,                 # T steps (maximum)
        start_values: Sequence[float],        # prices at t=0 (length K)
        initial_inventory: float = 1.0,       # Q
    ):
        if len(generators) < 1:
            raise ValueError("ExecutionTask requires at least one generator (venue).")
        super().__init__(generator=generators, agent=agent, total_movements=total_movements)

        self.generators: List[Any] = list(generators)
        self.K: int = len(self.generators)

        if len(start_values) != self.K:
            raise ValueError("start_values length must match number of generators.")
        self.values = np.asarray(start_values, dtype=float)
        self.values_prev = self.values.copy()

        self.initial_inventory = float(initial_inventory)
        self.q = self.initial_inventory

        self.cost_cum = 0.0
        self.cost_development: List[float] = [self.cost_cum]
        self.inventory_development: List[float] = [self.q]
        # self.log is initialized by Task

    # ---------- one step ----------
    def play_turn(self, step: int) -> None:
        # 1) Agent observes current state BEFORE acting
        remaining_time = self.total_movements - (step - 1)
        state = {
            "values": self.values.copy(),                  # current values across generators
            "remaining_inventory": float(self.q),          # scalar remaining resource
            "remaining_time": int(remaining_time),         # steps left incl. this one
            "cost_cum": float(self.cost_cum),              # cumulative *execution cost* so far
        }
        if hasattr(self.agent, "observe"):
            self.agent.observe(state)

        # 2) Agent proposes absolute order sizes (non-negative). Enforce feasibility.
        K = len(self.generators)
        raw = self.agent.place_bet() if hasattr(self.agent, "place_bet") else np.zeros(K, dtype=float)
        x = np.asarray(raw, dtype=float).reshape(-1)
        if x.size != K:
            x = np.zeros(K, dtype=float)
        x = np.clip(x, 0.0, None)

        remain = float(self.q)
        x_sum = float(x.sum())

        if step < self.total_movements:
            # Cap to remaining on intermediate steps
            if x_sum > remain and x_sum > 0.0:
                x = x * (remain / x_sum)
        else:
            # Final step: force liquidation if anything remains
            if remain > 0.0:
                if x_sum > 0.0:
                    x = x * (remain / x_sum)
                else:
                    # Neutral fallback: split remaining evenly
                    x = np.full(K, remain / K, dtype=float)
            else:
                x = np.zeros(K, dtype=float)

        # 3) Advance values (produce next-step values)
        new_values = np.array(
            [gen.generate_value(self.values[i]) for i, gen in enumerate(self.generators)],
            dtype=float
        )

        # 4) Execution cost ONLY (cash spent/consumed this step): next values • x
        step_cost = float(np.dot(new_values, x)) #  Agent acts on t, executes at t+1.
        self.cost_cum += step_cost  # cumulative execution cost

        # 5) Update inventory and value state
        self.q = max(0.0, self.q - float(x.sum()))
        values_prev = self.values
        self.values = new_values

        # 6) Log step
        self.log.append({
            "t": step,
            "values_prev": values_prev.copy(),   # pre-trade values used for pricing this step
            "values": new_values.copy(),         # next values after generators advance
            "x": x.copy(),                       # executed vector this step
            "q_remaining": float(self.q),        # remaining after execution
            "step_cost": step_cost,              # execution cost this step (cash)
            "cost_cum": float(self.cost_cum),    # cumulative execution cost
        })
        self.cost_development.append(self.cost_cum)
        self.inventory_development.append(self.q)


    # ---------- full run ----------
    def play_game(self) -> float:
        for step in range(1, self.total_movements + 1):
            if self.q <= 1e-12:  # fully executed
                break
            self.play_turn(step)
        # Return negative cost so higher is better (reward-style)
        return -self.cost_cum
