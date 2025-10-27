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

---

### ✅ Prediction Task (`tasks.py`)

- Simulates a sequential game between a **time series generator** and an **agent**
- Handles:
  - Time series generation (requires a generator with `.generate_value()`)
  - Agent action sampling via `.place_bet()`
  - Reward assignment (+1 correct / -1 incorrect)
  - Step logging and cumulative reward tracking

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


## Acknowledgments

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Research Executive Agency (REA). Neither the European Union nor the granting authority can be held responsible for them.

![EU Logo](images/eu_funded_logo.jpg)

## License

MIT — see LICENSE.