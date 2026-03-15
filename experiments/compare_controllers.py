import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

# Add the root directory to the python path using pathlib
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from controllers.ml_controller import MLController
from controllers.pid_controller import PIDController
from simulator.furnace_model import Furnace


def get_setpoint(step: int) -> float:
    """Return a piecewise setpoint profile for the closed-loop test."""
    if step < 150:
        return 120.0
    if step < 300:
        return 180.0
    if step < 400:
        return 90.0
    return 160.0


def run_simulation(controller, steps: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute one closed-loop simulation for a single controller."""
    furnace = Furnace()

    # Pre-allocate arrays to keep the simulation loop simple and efficient.
    history_temp = np.zeros(steps)
    history_power = np.zeros(steps)
    history_setpoint = np.zeros(steps)

    current_temp = furnace.reset()
    if hasattr(controller, "reset"):
        controller.reset()

    for t in range(steps):
        setpoint = get_setpoint(t)

        power = controller.control(target=setpoint, current=current_temp)
        current_temp = furnace.step(power=power)

        history_temp[t] = current_temp
        history_power[t] = power
        history_setpoint[t] = setpoint

    return history_temp, history_power, history_setpoint


def compute_metrics(history_temp: np.ndarray, history_setpoint: np.ndarray) -> dict[str, float]:
    """Compute a few simple tracking metrics for controller comparison."""
    abs_error = np.abs(history_setpoint - history_temp)
    steady_state_slice = slice(int(0.8 * len(abs_error)), None)

    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean((history_setpoint - history_temp) ** 2))),
        "steady_state_error": float(np.mean(abs_error[steady_state_slice])),
    }


def plot_results(
    pid_temp: np.ndarray,
    pid_power: np.ndarray,
    ml_temp: np.ndarray,
    ml_power: np.ndarray,
    history_setpoint: np.ndarray,
) -> None:
    """Plot both controllers on the same figures."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(pid_temp, label='PID Temperature', color='tab:blue')
    ax1.plot(ml_temp, label='ML Temperature', color='tab:green')
    ax1.plot(history_setpoint, label='Setpoint', color='tab:red', linestyle='--')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Furnace Temperature Control: PID vs ML')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(pid_power, label='PID Power', color='tab:orange')
    ax2.plot(ml_power, label='ML Power', color='tab:purple')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Power (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    """Pretty-print controller metrics."""
    print(f"{label} metrics:")
    print(f"  MAE: {metrics['mae']:.3f}")
    print(f"  RMSE: {metrics['rmse']:.3f}")
    print(f"  Steady-state error: {metrics['steady_state_error']:.3f}")


if __name__ == "__main__":
    model_path = root_dir / "results" / "furnace_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run training/train_model.py first."
        )

    model = joblib.load(model_path)

    pid_controller = PIDController(kp=2.0, ki=0.1, kd=0.5)
    ml_controller = MLController(model=model, n_candidates=101, power_penalty=0.03)

    pid_temp, pid_power, setpoint = run_simulation(pid_controller, steps=500)
    ml_temp, ml_power, _ = run_simulation(ml_controller, steps=500)

    pid_metrics = compute_metrics(pid_temp, setpoint)
    ml_metrics = compute_metrics(ml_temp, setpoint)

    print_metrics("PID", pid_metrics)
    print_metrics("ML", ml_metrics)

    plot_results(pid_temp, pid_power, ml_temp, ml_power, setpoint)