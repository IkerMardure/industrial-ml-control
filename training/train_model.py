import argparse
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS: Final[list[str]] = ["temp_t_minus_2", "temp_t_minus_1", "current_temp","power_t_minus_2","power_t_minus_1", "power"]
TARGET_COLUMN: Final[str] = "next_temp"


def train_furnace_model(data_path: Path, model_output_path: Path) -> None:
    print(f"Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # Load the dataset and validate the required schema.
    df = pd.read_csv(data_path)
    missing_columns = [column for column in [*FEATURE_COLUMNS, TARGET_COLUMN] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Keep training features aligned with the runtime ML controller.
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    # Split data into train and test sets for an unbiased evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    print("Training Random Forest model...")
    # Use a moderately larger forest to improve robustness without overcomplicating the script.
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    # Report multiple metrics to understand both average and squared error behavior.
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Persist the trained model so the controller can load it later.
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    
    print(f"Model saved to: {model_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML model for furnace physics.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("results/training_data.csv"),
        help="Input CSV dataset path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/furnace_model.joblib"),
        help="Output path for the trained model.",
    )
    args = parser.parse_args()

    train_furnace_model(data_path=args.data, model_output_path=args.output)


if __name__ == "__main__":
    main()