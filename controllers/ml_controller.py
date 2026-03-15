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
		self.last_power = 0.0

	def reset(self) -> None:
		self.last_power = 0.0

	def control(self, target: float, current: float) -> float:
		low, high = self.power_bounds
		candidates = np.linspace(low, high, self.n_candidates)

		features = np.column_stack([
			np.full_like(candidates, fill_value=current, dtype=float),
			candidates,
		])
		predicted_temp = self.model.predict(features)

		tracking_cost = np.abs(predicted_temp - target)
		smoothness_cost = self.power_penalty * np.abs(candidates - self.last_power)
		total_cost = tracking_cost + smoothness_cost

		best_idx = int(np.argmin(total_cost))
		best_power = float(candidates[best_idx])
		self.last_power = best_power
		return best_power
