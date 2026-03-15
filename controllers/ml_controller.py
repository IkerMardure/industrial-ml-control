from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class _PredictModel(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class MLController:
    """Multi-step Model Predictive Controller using Random Shooting."""

    model: _PredictModel
    power_bounds: tuple[float, float] = (0.0, 100.0)
    horizon: int = 5            # Pasos hacia el futuro a simular
    n_samples: int = 1000       # Rutas aleatorias a evaluar por iteración
    penalty_d1: float = 0.05    # Penalización por cambiar la potencia (suavidad)
    penalty_d2: float = 0.05    # Penalización por cambios bruscos de dirección (chattering)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.temp_t1: float | None = None
        self.temp_t2: float | None = None
        self.power_t1: float = 0.0
        self.power_t2: float = 0.0

    def control(self, target: float, current: float) -> float:
        if self.temp_t1 is None or self.temp_t2 is None:
            self.temp_t1 = current
            self.temp_t2 = current

        low, high = self.power_bounds

        # 1. Generar K secuencias aleatorias de acciones de longitud H
        action_seqs = np.random.uniform(low, high, size=(self.n_samples, self.horizon))

        # Matrices para mantener el estado de las K simulaciones simultáneas
        sim_temp_t2 = np.full(self.n_samples, self.temp_t2)
        sim_temp_t1 = np.full(self.n_samples, self.temp_t1)
        sim_curr_temp = np.full(self.n_samples, current)
        sim_power_t2 = np.full(self.n_samples, self.power_t2)
        sim_power_t1 = np.full(self.n_samples, self.power_t1)

        total_cost = np.zeros(self.n_samples)

        # 2. Simular hacia adelante y calcular el coste de seguimiento (Tracking Cost)
        for h in range(self.horizon):
            current_actions = action_seqs[:, h]

            features = np.column_stack([
                sim_temp_t2,
                sim_temp_t1,
                sim_curr_temp,
                sim_power_t2,
                sim_power_t1,
                current_actions
            ])

            next_temps = self.model.predict(features)

            # Sumar el error absoluto de este paso futuro al coste total
            total_cost += np.abs(next_temps - target)

            # Desplazar la memoria para el siguiente paso del horizonte
            sim_temp_t2 = sim_temp_t1
            sim_temp_t1 = sim_curr_temp
            sim_curr_temp = next_temps
            sim_power_t2 = sim_power_t1
            sim_power_t1 = current_actions

        # 3. Calcular las penalizaciones de control
        # Añadimos las acciones pasadas reales para calcular las derivadas correctamente
        past_actions = np.column_stack([
            np.full(self.n_samples, self.power_t2),
            np.full(self.n_samples, self.power_t1)
        ])
        full_actions = np.hstack([past_actions, action_seqs])

        # Penalización D1: |u_t - u_{t-1}|
        diff1 = np.diff(full_actions, n=1, axis=1)
        cost_d1 = np.abs(diff1).sum(axis=1) * self.penalty_d1

        # Penalización D2: |u_t - 2u_{t-1} + u_{t-2}|
        diff2 = np.diff(full_actions, n=2, axis=1)
        cost_d2 = np.abs(diff2).sum(axis=1) * self.penalty_d2

        total_cost += cost_d1 + cost_d2

        # 4. Seleccionar la mejor ruta y extraer solo la primera acción
        best_idx = int(np.argmin(total_cost))
        best_action = float(action_seqs[best_idx, 0])

        # 5. Actualizar la memoria real del controlador
        self.temp_t2 = self.temp_t1
        self.temp_t1 = current
        self.power_t2 = self.power_t1
        self.power_t1 = best_action

        return best_action