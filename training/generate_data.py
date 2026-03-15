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

	for _ in range(n_episodes):
		ambient = float(rng.uniform(15.0, 30.0))
		initial_temp = float(rng.uniform(ambient - 2.0, ambient + 2.0))
		target = float(rng.uniform(60.0, 220.0))

		furnace = Furnace(ambient_temp=ambient)
		furnace.reset()
		furnace.current_temp = initial_temp
		current_temp = initial_temp
		prev_power = float(rng.uniform(0.0, 100.0))

		for _ in range(steps_per_episode):
			# Randomized but target-aware policy to collect diverse transitions.
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
					"current_temp": current_temp,
					"power": power,
					"next_temp": next_temp,
					"target_temp": target,
				}
			)

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
