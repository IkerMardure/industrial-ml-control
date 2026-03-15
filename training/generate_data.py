from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this file directly with `python training/generate_data.py`.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from simulator import Furnace


def generate_dataset(
    n_episodes: int = 120,
    steps_per_episode: int = 250,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []

    for episode in range(n_episodes):
        # La mitad de los episodios tendrán eventos extremos y "domain randomization"
        is_advanced_episode = episode < (n_episodes // 2)

        ambient = float(rng.uniform(15.0, 30.0))
        initial_temp = float(rng.uniform(ambient - 2.0, ambient + 2.0))
        target = float(rng.uniform(60.0, 220.0))
        
        if is_advanced_episode:
            heating = float(rng.uniform(0.4, 0.6))
            cooling = float(rng.uniform(0.03, 0.07))
            furnace = Furnace(
                ambient_temp=ambient, 
                heating_rate=heating, 
                cooling_rate=cooling
            )
        else:
            furnace = Furnace(ambient_temp=ambient)

        furnace.reset()
        furnace.current_temp = initial_temp
        
        current_temp = initial_temp
        prev_power = float(rng.uniform(0.0, 100.0))
        
        # Variables de memoria temporal
        temp_t2 = current_temp
        temp_t1 = current_temp
        power_t2 = 0.0
        power_t1 = 0.0
        
        override_steps = 0
        override_power = 0.0

        for step in range(steps_per_episode):
            if is_advanced_episode:
                # Cambios bruscos de setpoint a mitad de episodio
                if step > 0 and step % 80 == 0:
                    if rng.random() < 0.5:
                        target = ambient
                    else:
                        target = float(rng.uniform(60.0, 220.0))

                # Lógica para inyectar eventos extremos
                if override_steps > 0:
                    power = override_power
                    override_steps -= 1
                else:
                    if rng.random() < 0.05:
                        event = rng.choice(["max_heat", "zero_heat"])
                        override_steps = int(rng.integers(10, 25))
                        override_power = 100.0 if event == "max_heat" else 0.0
                        power = override_power
                    else:
                        power = (
                            0.55 * prev_power
                            + 0.30 * rng.uniform(0.0, 100.0)
                            + 0.15 * (target - current_temp)
                        )
            else:
                # Política normal
                power = (
                    0.55 * prev_power
                    + 0.30 * rng.uniform(0.0, 100.0)
                    + 0.15 * (target - current_temp)
                )
            
            power = float(np.clip(power, 0.0, 100.0))
            next_temp = furnace.step(power=power)

            rows.append(
                {
                    "ambient_temp": ambient,
                    "temp_t_minus_2": temp_t2,
                    "temp_t_minus_1": temp_t1,
                    "current_temp": current_temp,
                    "power_t_minus_2": power_t2,
                    "power_t_minus_1": power_t1,
                    "power": power,
                    "next_temp": next_temp,
                    "target_temp": target,
                }
            )

            # Shift de memoria
            temp_t2 = temp_t1
            temp_t1 = current_temp
            power_t2 = power_t1
            power_t1 = power
            current_temp = next_temp
            prev_power = power

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate furnace simulation dataset.")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/training_data.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    df = generate_dataset(
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Dataset generated: {len(df)} rows")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()