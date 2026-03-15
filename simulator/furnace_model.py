from dataclasses import dataclass
import numpy as np

@dataclass
class Furnace:
    """Industrial furnace physical simulation model."""

    ambient_temp: float = 20.0
    heating_rate: float = 0.5
    cooling_rate: float = 0.05
    noise_std: float = 0.5

    def __post_init__(self) -> None:
        self.current_temp: float = self.ambient_temp

    def reset(self) -> float:
        """Resets the furnace to ambient temperature."""
        self.current_temp = self.ambient_temp
        return self.current_temp

    def step(self, power: float) -> float:
        """
        Advances the simulation one time step.
        
        Args:
            power: Applied power percentage.
            
        Returns:
            Measured temperature with simulated sensor noise.
        """
        actual_power = float(np.clip(power, 0.0, 100.0))
        
        heating = actual_power * self.heating_rate
        cooling = (self.current_temp - self.ambient_temp) * self.cooling_rate
        
        self.current_temp += heating - cooling
        
        noise = float(np.random.normal(loc=0.0, scale=self.noise_std))
        measured_temp = self.current_temp + noise
        
        return measured_temp