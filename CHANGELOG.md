# Changelog
All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## Unreleased
*Add upcoming changes here before creating the next release.*

### Added
- (none yet)

### Changed
- (none yet)

### Deprecated
- (none yet)

### Removed
- (none yet)

### Fixed
- (none yet)

---
## v0.2.5 — 2025-10-28

### Added
- Introduced the **Execution Task** environment for evaluating agents under finite-inventory execution settings.
- Added **test suite** covering execution task.
- Added **examples** demonstrating basic usage of prediction, allocation and execution task.

---

## v0.2.4 — 2025-10-28

### Added
- **General `Task` Base Class**
  - Introduced a shared abstract interface and lifecycle (`play_turn`, `play_game`).
  - Centralized shared attributes (`agent`, `generator(s)`, `total_movements`, `log`).

- **`AllocationTask`**
  - Multi-generator, long-only allocation environment.
  - Supports:
    - Vector observations to agent (`observe(values)`).
    - Allocation vector decisions from agent (`place_bet()` → weights).
    - Simplex projection for valid allocation enforcement.
    - Budget updating with optional turnover cost penalty.
  - Full step-level structured logging and budget progression tracking.

### Changed
- **PredictionTask refactor**
  - Now subclasses the general `Task` class for consistent interface structure.
  - Logging standardized into **dict-based per-step log** (self-describing fields).
  - Legacy `bet_log` converted into deprecated compatibility wrapper.

### Tests
- **General Task Class Tests**
  - Validated that subclasses must implement required abstract methods.
  - Confirmed initialization invariants and logging container availability.

- **AllocationTask Test Suite**
  - Budget behavior across rising, falling, and mixed-value environments.
  - Observation order correctness (`agent.observe` called before each step).
  - Log schema and dimensionality checks.
  - Simplex projection enforcement tests:
    - Negative / non-summing weights corrected.
    - All-zero proposals → uniform allocation.
  - Turnover cost penalty correctness.
  - Edge case handling:
    - Zero steps → no state evolution.
    - Incorrect `start_values` length → raises `ValueError`.
    - Fewer than two generators → raises `ValueError`.

### Deprecated
- No new deprecations introduced.

---

## v0.2.3 — 2025-10-27
### Changed
- Updated the README to replace **`observer`** with **`agent`** for consistent terminology across code, documentation, and user-facing examples.

---

## v0.2.2 — 2025-10-27
### Changed
- Renamed core environment class from **`BettingGame`** to **`PredictionTask`** for clearer semantic meaning.
- Updated internal references and documentation to use `PredictionTask` consistently.

### Deprecated
- **`BettingGame`** is now deprecated and will be removed in **v0.3.0**.
- Instantiating `BettingGame` emits a `DeprecationWarning` and forwards all arguments to `PredictionTask`.

```python
# Old (deprecated)
game = BettingGame(generator, observer, total_movements=100)

# New (preferred)
game = PredictionTask(generator, observer, total_movements=100)
```
### Added
- Comprehensive test suite for `PredictionTask`, including:
  - Parametrized reward behavior tests for rising/falling series and agent prediction correctness.
  - Verification that agents observe the previous value before placing a bet (correct interaction order).
  - Consistency checks for `last_value` updates after each step.
  - Validation of logging behavior:
    - `reward_development` tracks cumulative reward correctly.
    - `bet_log` stores `(step, value, bet, received_reward)` tuples in correct format.
  - Edge case handling:
    - `total_movements = 0` (no steps performed).
    - Flat/no-change generators (ties penalize the agent under current rules).
    - Custom `start_value` forwarding.
  - Deprecation wrapper tests ensuring:
    - Instantiating `BettingGame` raises a `DeprecationWarning`.
    - The wrapper returns a `PredictionTask` instance.
    - Parameters (e.g., `start_value`) are forwarded correctly.

### Test Summary

27 passed in 3.57s.