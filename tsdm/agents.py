

# ========================================================================
# General Imports
# ========================================================================


# General imports
from abc import ABC, abstractmethod
import numpy as np

# SGDClassifier imports
from sklearn.linear_model import SGDClassifier

# DQN imports
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


# ========================================================================
# Abstract Base Class for Agents
# ========================================================================


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
        raise NotImplementedError("Subclasses must implement the place_bet() method")

    def reset(self):
        """Clears observed values — useful between game runs."""
        self.observed_values = []

# ========================================================================
# Zero-Intelligence & Heuristic Agents
# ========================================================================


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

# ========================================================================
# Statistical Agents
# ========================================================================


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

# ========================================================================
# Supervised Learning Agent (SGD Classifier)
# ========================================================================


class SGDClassifierAgent(Agent):
    """
    SGDClassifierAgent

    Online-learning agent using SGDClassifier.
    - Trains on windowed past observations to predict next movement.
    - Avoids lookahead bias by always training on past values before prediction.
    """

    def __init__(self, window_size=50):
        super().__init__()
        self.window_size = window_size
        self.model = SGDClassifier(loss='hinge', random_state=42)
        self.has_been_fitted = False

    def place_bet(self):
        """
        Predicts the next movement using the most recent window.
        Returns 0 if not enough data.
        """
        if len(self.observed_values) < self.window_size:
            return 0  # Default prediction

        X = np.array(self.observed_values[-self.window_size:]).reshape(1, -1)
        if self.has_been_fitted:
            prediction = self.model.predict(X)[0]
            return int(prediction)
        else:
            return 0  # Predict default until model is fitted

    def observe(self, value):
        """
        Observes new value and trains model on past window if possible.
        """
        super().observe(value)

        # Train only if enough past data (window + target value)
        if len(self.observed_values) >= self.window_size + 1:
            X_train = np.array(self.observed_values[-self.window_size - 1:-1]).reshape(1, -1)
            y_train = [1 if self.observed_values[-1] > self.observed_values[-2] else 0]

            if not self.has_been_fitted:
                self.model.partial_fit(X_train, y_train, classes=[0, 1])
                self.has_been_fitted = True
            else:
                self.model.partial_fit(X_train, y_train)

    def reset(self):
        super().reset()
        self.model = SGDClassifier(loss='hinge', random_state=42)
        self.has_been_fitted = False

# ========================================================================
# Deep Q-Network Components
# ========================================================================


class DQNetwork(nn.Module):
    """
    DQNetwork

    Simple feedforward neural network for approximating Q-values.
    - Input: state_size features (past observed values).
    - Output: Q-values for each possible action (0 = down, 1 = up).
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Two actions: down (0), up (1)
        )

    def forward(self, x):
        return self.model(x)


class DQNAgent(Agent):
    """
    DQNAgent

    A Deep Q-Learning Agent for the betting game.
    - Learns to predict the expected reward (Q-value) of betting up or down.
    - Trains incrementally using experience replay.

    Attributes:
        state_size (int): Number of past observed values forming the state.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Exploration rate for ε-greedy action selection.
        epsilon_min (float): Minimum exploration rate after decay.
        epsilon_decay (float): Multiplicative decay factor for epsilon.
        batch_size (int): Number of samples used per training step.
        memory (deque): Experience replay buffer.
        device (torch.device): CPU or GPU.
        model (DQNetwork): Neural network approximating Q-values.
        optimizer (torch.optim): Optimizer for training.
        loss_fn (nn.Module): Loss function for Q-learning updates.
        last_state (np.array): Last state used for training.
        last_action (int): Last action taken.
    """

    def __init__(self, state_size=10, gamma=0.95, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.995, lr=1e-3, batch_size=32, memory_size=1000):
        super().__init__()
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQNetwork(input_dim=state_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.last_state = None
        self.last_action = None

    def place_bet(self):
        """
        Decides on a betting action (0 or 1) using ε-greedy strategy.
        - If insufficient history, defaults to 0.
        - Stores the state and action for learning during observe().
        Returns:
            int: Chosen action (0 = down, 1 = up).
        """
        if len(self.observed_values) < self.state_size:
            return 0  # Cold start action

        state = np.array(self.observed_values[-self.state_size:], dtype=np.float32)
        self.last_state = state

        if np.random.rand() <= self.epsilon:
            action = random.choice([0, 1])  # Explore
        else:
            with torch.no_grad():
                q_vals = self.model(torch.tensor(state).unsqueeze(0).to(self.device))
                action = torch.argmax(q_vals).item()  # Exploit

        self.last_action = action
        return action

    def observe(self, value):
        """
        Observes a new value, updates memory, and triggers training.
        - Stores experience tuple in the replay buffer.
        - Calculates immediate reward based on previous action.
        - Trains the DQN on a random batch if enough samples exist.
        """
        super().observe(value)

        if len(self.observed_values) < self.state_size + 1:
            return  # Not enough data for state transition

        next_state = np.array(self.observed_values[-self.state_size:], dtype=np.float32)

        # Only update memory if last_state and last_action exist and are valid
        if self.last_state is not None and self.last_action is not None:
            if len(self.last_state) == self.state_size and len(next_state) == self.state_size:
                # Reward: +1 if prediction correct, else -1
                if self.last_action == 1 and self.observed_values[-1] > self.observed_values[-2]:
                    reward = 1
                elif self.last_action == 0 and self.observed_values[-1] < self.observed_values[-2]:
                    reward = 1
                else:
                    reward = -1

                self.memory.append((self.last_state, self.last_action, reward, next_state))

        # Train network on batch
        self._train()


    def _train(self):
        """
        Samples a batch from replay memory and performs a Q-learning update.
        - Computes current Q-values for the selected actions.
        - Computes target Q-values using Bellman equation.
        - Applies mean squared error loss between current and target.
        - Decays epsilon after each training step.
        """
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to train

        # Sample random batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        # Convert data to tensors
        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)

        # Compute predicted Q-values for the actions actually taken
        current_q = self.model(states).gather(1, actions)

        # Compute max Q-values for next states (target Q-learning step)
        next_q = self.model(next_states).max(1)[0].detach().unsqueeze(1)

        # Compute target values
        target_q = rewards + self.gamma * next_q

        # Compute loss between current Q and target Q
        loss = self.loss_fn(current_q, target_q)

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay for exploration-exploitation tradeoff
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def reset(self):
        """
        Resets internal state between episodes.
        - Clears observed values and last state/action.
        """
        super().reset()
        self.last_state = None
        self.last_action = None





