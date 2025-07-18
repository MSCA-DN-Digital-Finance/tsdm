import numpy as np


class BettingGame:
    """
    Simulates a sequential betting game between a generator and an observer (agent).

    - The generator produces a new value at each time step.
    - The observer places a bet predicting whether the next value will be higher or lower.
    - The game rewards the observer for correct predictions and penalizes for incorrect ones.

    Attributes:
        generator: An object with a `generate_value()` method that produces the next value.
        observer: An agent with `observe()` and `place_bet()` methods.
        total_movements: Total number of time steps to play.
        last_value: The last generated value (starts at `start_value`).
        reward: Cumulative reward (increases or decreases based on prediction accuracy).
        reward_development: Array tracking cumulative reward over time.
        bet_log: History of (step, value, bet, reward) tuples for analysis.
    """

    def __init__(self, generator, observer, total_movements, start_value=0):
        self.generator = generator
        self.observer = observer
        self.total_movements = total_movements
        self.last_value = start_value
        self.reward = 0
        self.reward_development = np.array([])
        self.bet_log = []  # Stores (step, value, bet, reward)

    def play_turn(self, step):
        """
        Plays a single turn of the game:
        - Generates a new value using the generator's `generate_value()`.
        - Observer places a bet (1 = up, 0 = down).
        - Calculates the received reward (+1 or -1).
        - Updates cumulative reward and logs the result.
        """
        value = self.generator.generate_value(self.last_value)
        bet = self.observer.place_bet()

        # Determine if the bet is correct and assign reward
        if (value > self.last_value and bet == 1) or (value < self.last_value and bet == 0):
            received_reward = 1
        else:
            received_reward = -1

        # Update cumulative reward
        self.reward += received_reward
        self.reward_development = np.append(self.reward_development, self.reward)

        # Log the result
        self.bet_log.append((step, value, bet, received_reward))

        # Update for next round
        self.last_value = value

    def play_game(self):
        """
        Plays the full game over the specified number of movements.
        The observer receives the latest movement before each turn.
        Returns the final cumulative reward.
        """
        for step in range(1, self.total_movements + 1):
            self.observer.observe(self.last_value)
            self.play_turn(step)

        return self.reward
