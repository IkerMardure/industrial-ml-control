from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class _PredictModel(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class MLController:
    """One-step predictive controller using a learned process model."""

    model: _PredictModel
    power_bounds: tuple[float, float] = (0.0, 100.0)
    n_candidates: int = 101
    power_penalty: float = 0.02

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # We use None to detect the very first step of an episode
        self.temp_t1: float | None = None
        self.temp_t2: float | None = None
        self.power_t1: float = 0.0
        self.power_t2: float = 0.0
        self.last_power: float = 0.0

    def control(self, target: float, current: float) -> float:
        # 1. Backfill memory on the very first step
        if self.temp_t1 is None or self.temp_t2 is None:
            self.temp_t1 = current
            self.temp_t2 = current

        low, high = self.power_bounds
        candidates = np.linspace(low, high, self.n_candidates)

        # 2. Build the feature matrix. 
        # IMPORTANT: The column order must exactly match the FEATURE_COLUMNS list 
        # you defined in your training/train_model.py script.
        features = np.column_stack([
            np.full_like(candidates, fill_value=self.temp_t2, dtype=float),
            np.full_like(candidates, fill_value=self.temp_t1, dtype=float),
            np.full_like(candidates, fill_value=current, dtype=float),
            np.full_like(candidates, fill_value=self.power_t2, dtype=float),
            np.full_like(candidates, fill_value=self.power_t1, dtype=float),
            candidates,
        ])
        
        predicted_temp = self.model.predict(features)

        tracking_cost = np.abs(predicted_temp - target)
        smoothness_cost = self.power_penalty * np.abs(candidates - self.last_power)
        total_cost = tracking_cost + smoothness_cost

        best_idx = int(np.argmin(total_cost))
        best_power = float(candidates[best_idx])
        
        # 3. Shift the memory window for the next step
        self.temp_t2 = self.temp_t1
        self.temp_t1 = current
        self.power_t2 = self.power_t1
        self.power_t1 = best_power
        self.last_power = best_power
        
        return best_power