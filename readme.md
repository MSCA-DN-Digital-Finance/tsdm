# TSDM — Time Series Decision-Making Benchmark (Alpha)

**TSDM** is a lightweight benchmarking framework for evaluating agents in sequential decision-making tasks based on temporal data.

It provides a clean, Gym-free setup with controlled environments and standardized agents — ideal for systematic experimentation in **time-series-only decision-making**.

---

## 📝 Overview

TSDM focuses on agents that observe a single temporal value at each step and must sequentially decide on a binary action (betting up or down).  
Performance is tracked through cumulative rewards based on correct or incorrect predictions.

---

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install tsdm-benchmark
```

---

## 📦 Current Components

### ✅ Core Agent Classes (`agents.py`)

- **Abstract Base Class**
  - `Agent` — defines the interface for all agents (`observe()`, `place_bet()`, `reset()`)

- **Baseline Heuristic Agents**
  - `AlwaysUpAgent` — Always predicts "up"
  - `RepeatLastMovementAgent` — Predicts based on last movement direction
  - `FrequencyBasedMajorityAgent` — Predicts based on majority of past movement directions

- **Statistical Agents**
  - `StaticMeanReversionAgent` — Bets on reversion to historical mean
  - `DynamicMeanReversionAgent` — Bets on mean reversion within a rolling time window

- **Learning Agents**
  - `SGDClassifierAgent` — Online learning using scikit-learn’s `SGDClassifier`
  - `DQNAgent` — Deep Q-Learning agent with experience replay (PyTorch-based)

### ✅ Task Environments (`tasks.py`)

This module defines general sequential decision-making tasks.  
Each task models an interaction between:

- **Generators**: produce evolving values over time  
- **Agent**: observes values and chooses an action  
- **Task**: applies update mechanics and logs outcomes


#### **Prediction Task**

A binary directional decision task.

**Process**
- A single generator produces a new value each step.
- The agent observes the previous value and predicts whether the next value will go up or down.
- The agent receives a positive reward for a correct prediction and a negative reward otherwise.

**Characteristics**
- Single evolving value stream
- Binary decision output from the agent
- Cumulative reward and full step-by-step logs are recorded
- Useful for evaluating pattern recognition, signal prediction, and directional inference



#### **Allocation Task**

A multi-source allocation task.

**Process**
- Multiple generators evolve in parallel, each producing their own value sequence.
- The agent observes the vector of current values and returns a vector of allocations.
- Allocations must be non-negative and sum to one (long-only simplex constraint).
- The budget evolves according to relative changes in the values weighted by the allocations.
- Optionally, changing allocations incurs turnover cost.

**Characteristics**
- Works with two or more generators simultaneously
- Agent chooses a proportional distribution rather than a binary decision
- Full step logging includes values, allocations, relative changes, turnover, and budget trajectory
- Useful for adaptive weighting, meta-learning, portfolio-style allocation, and combining experts


#### **Execution Task**

A finite-inventory execution and liquidation task.

**Process**
- Multiple generators evolve in parallel, each producing their own value sequence over time.
- The agent starts with a fixed **inventory** (e.g., units of an asset) that must be fully executed by the end of the task.
- At each step, the agent observes the current values and chooses **absolute execution amounts** for each generator (non-negative real values).
- Executing inventory incurs a **cost**, determined by the *next-step* values generated after the action is chosen.
- If any inventory remains on the **final step**, it is automatically **liquidated evenly** across generators.

**Characteristics**
- Multiple generators with evolving values
- Agent outputs *absolute quantities* (not proportions)
- Strict finite-inventory constraint: inventory only decreases
- Execution cost accumulates over time based on future (t+1) values
- Full logs include:
  - pre-execution values
  - post-execution next-step values
  - execution vector per step
  - remaining inventory trajectory
  - step-by-step and cumulative cost
- Useful for studying:
  - optimal liquidation timing
  - execution strategies under uncertainty
  - anticipation of future value movement
  - allocation vs. patience trade-offs

---

## 🎲 Example Usage

```python
from tsdm.agents import AlwaysUpAgent
from tsdm.tasks import PredictionTask
from tsg.generators import LinearTrendGenerator  # Assuming you have installed tsg-lib

agent = AlwaysUpAgent()
generator = LinearTrendGenerator(slope=0.1)
game = PredictionTask(generator=generator, agent=agent, total_movements=1000)

final_reward = game.play_game()
print(f"Final cumulative reward: {final_reward}")
```
More detailed demonstrations for each task can be found in the `examples` folder.

---

## Acknowledgments

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Research Executive Agency (REA). Neither the European Union nor the granting authority can be held responsible for them.

![EU Logo](images/eu_funded_logo.jpg)

---

## License

MIT — see LICENSE.