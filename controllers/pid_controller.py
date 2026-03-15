from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PIDController:
	"""Classic PID controller with integral and output limits."""

	kp: float
	ki: float
	kd: float
	dt: float = 1.0
	output_limits: tuple[float, float] = (0.0, 100.0)
	integral_limits: tuple[float, float] = (-1_000.0, 1_000.0)

	def __post_init__(self) -> None:
		self.integral = 0.0
		self.prev_error = 0.0

	def reset(self) -> None:
		self.integral = 0.0
		self.prev_error = 0.0

	def control(self, target: float, current: float) -> float:
		error = float(target - current)
		self.integral += error * self.dt

		i_low, i_high = self.integral_limits
		self.integral = float(np.clip(self.integral, i_low, i_high))

		derivative = (error - self.prev_error) / self.dt
		self.prev_error = error

		output = self.kp * error + self.ki * self.integral + self.kd * derivative
		low, high = self.output_limits
		return float(np.clip(output, low, high))
