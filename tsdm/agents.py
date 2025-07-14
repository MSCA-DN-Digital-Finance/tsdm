from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    """
    Abstract base class for all betting agents.

    - Observes values sequentially via `observe(value)`.
    - Makes predictions (bets) via `place_bet()`.
    - Stores observed values for possible use by subclasses.
    """

    def __init__(self):
        self.observed_values = []

    def observe(self, value):
        """Receives a new observed value from the environment."""
        self.observed_values.append(value)

    @abstractmethod
    def place_bet(self):
        """Returns the agent's bet (e.g., 0 = down, 1 = up). Must be implemented by subclasses."""
        pass

    def reset(self):
        """Clears observed values — useful between game runs."""
        self.observed_values = []



class AlwaysUpAgent(Agent):
    """
    AlwaysUpAgent

    This agent always bets "up" (1), regardless of the observed values.
    It serves as a zero-intelligence baseline agent for benchmarking.

    Inherits:
        - observe(value): Stores observed values (unused here).
        - reset(): Clears observed values between episodes.
    
    Methods:
        - place_bet(): Always returns 1.
    """

    def place_bet(self):
        """
        Returns:
            int: Always returns 1 (predict up).
        """
        return 1
    


class RepeatLastMovementAgent(Agent):
    """
    RepeatLastMovementAgent

    A fixed-rule agent that always repeats the direction of the last observed movement.
    - If the last price increased, it bets up (1).
    - If the last price decreased, it bets down (0).

    This is a simple momentum-based heuristic.
    """
    def place_bet(self):
        if len(self.observed_values) < 2:
            return 0  # Or 1, depending on your default choice for cold start

        previous = self.observed_values[-1]
        previous_previous = self.observed_values[-2]

        return 1 if previous >= previous_previous else 0


class FrequencyBasedMajorityAgent(Agent):
    """
    FrequencyBasedMajorityAgent

    This agent tracks the frequency of "up" and "down" movements observed so far.
    - Bets on the direction that occurred more frequently.
    - Uses a simple majority vote heuristic.
    
    Attributes:
        higher_count (int): Number of times the observed value increased.
        lower_count (int): Number of times the observed value decreased.
        last_value (float): Stores the last observed value for comparison.
    """

    def __init__(self):
        super().__init__()
        self.higher_count = 0
        self.lower_count = 0
        self.last_value = None

    def observe(self, value):
        """
        Observes a new value and updates frequency counts.
        Compares it with the last observed value to increment counters.
        """
        super().observe(value)
        if self.last_value is not None:
            if value > self.last_value:
                self.higher_count += 1
            elif value < self.last_value:
                self.lower_count += 1
        self.last_value = value

    def place_bet(self):
        """
        Places a bet based on the majority of observed movements.
        - If more downward movements observed, bets down (0).
        - Otherwise, bets up (1).
        """
        if self.higher_count < self.lower_count:
            return 0  # Predict down
        else:
            return 1  # Predict up

    def reset(self):
        """
        Resets the agent's internal state and observation history.
        """
        super().reset()
        self.higher_count = 0
        self.lower_count = 0
        self.last_value = None


class StaticMeanReversionAgent(Agent):
    """
    StaticMeanReversionAgent

    This agent applies a static mean reversion strategy using the full history of observed values.
    - Computes the mean of all observed values seen so far.
    - Bets that the next value will revert toward this historical mean:
      - If the last observed value is below the mean, bets up (1).
      - If the last observed value is above the mean, bets down (0).
    
    Attributes:
        Inherits observed_values and reset() from Agent.
    """

    def place_bet(self):
        """
        Places a bet based on mean reversion over the full observation history.
        Returns 0 (down) if no observations have been made yet.
        """
        if len(self.observed_values) == 0:
            return 0  # Default action before any data is observed

        mean = np.mean(self.observed_values)
        last_observed = self.observed_values[-1]

        return 1 if last_observed < mean else 0
    

class DynamicMeanReversionAgent(Agent):
    """
    DynamicMeanReversionAgent

    This agent follows a dynamic (windowed) mean reversion strategy:
    - Calculates the mean of the last `time_window` observed values.
    - Bets that the next value will revert toward this rolling mean.
      - If the last observed value is below the windowed mean, bets up (1).
      - If the last observed value is above the windowed mean, bets down (0).
    
    Attributes:
        time_window (int): The size of the observation window used to compute the mean.
        Inherits observed_values and reset() from Agent.
    """

    def __init__(self, time_window):
        super().__init__()
        self.time_window = time_window

    def place_bet(self):
        """
        Places a bet based on mean reversion over the latest time window.
        Returns 0 (down) if not enough data is observed yet.
        """
        if len(self.observed_values) < self.time_window:
            return 0  # Default action before enough data is collected

        window_values = self.observed_values[-self.time_window:]
        mean = np.mean(window_values)
        last_observed = self.observed_values[-1]

        return 1 if last_observed < mean else 0