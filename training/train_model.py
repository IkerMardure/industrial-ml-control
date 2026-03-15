from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

# Las características deben coincidir exactamente con las del MLController
FEATURE_COLUMNS: Final[list[str]] = [
    "temp_t_minus_2",
    "temp_t_minus_1",
    "current_temp",
    "power_t_minus_2",
    "power_t_minus_1",
    "power",
]
TARGET_COLUMN: Final[str] = "next_temp"


def evaluate_by_regime(df_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Evalúa el rendimiento del modelo segmentado por el estado físico del horno."""
    
    # Definir las máscaras lógicas para cada régimen
    near_setpoint = np.abs(df_test["current_temp"] - df_test["target_temp"]) <= 2.0
    heating = (df_test["target_temp"] > df_test["current_temp"] + 2.0) & (df_test["power"] > 50.0)
    cooling = (df_test["target_temp"] < df_test["current_temp"] - 2.0) & (df_test["power"] < 10.0)

    print("\n--- Error Absoluto Medio (MAE) por Régimen ---")
    
    if near_setpoint.sum() > 0:
        mae = mean_absolute_error(y_true[near_setpoint], y_pred[near_setpoint])
        print(f"Near Setpoint (Mantenimiento) [{near_setpoint.sum()} samples]: {mae:.4f} °C")
        
    if heating.sum() > 0:
        mae = mean_absolute_error(y_true[heating], y_pred[heating])
        print(f"Heating (Calentamiento activo) [{heating.sum()} samples]: {mae:.4f} °C")
        
    if cooling.sum() > 0:
        mae = mean_absolute_error(y_true[cooling], y_pred[cooling])
        print(f"Cooling (Enfriamiento pasivo) [{cooling.sum()} samples]: {mae:.4f} °C")


def train_furnace_model(data_path: Path, model_output_path: Path) -> None:
    print(f"Cargando datos desde: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {data_path}")

    df = pd.read_csv(data_path)
    
    # Dividir secuencialmente (shuffle=False) para evitar fuga de datos temporales.
    # Así entrenamos con los primeros N episodios y validamos con los últimos.
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)

    X_train = df_train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = df_train[TARGET_COLUMN].to_numpy(dtype=float)
    X_test = df_test[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = df_test[TARGET_COLUMN].to_numpy(dtype=float)

    print(f"Muestras de entrenamiento: {len(X_train)}")
    print(f"Muestras de validación: {len(X_test)}")
    
    print("\nIniciando búsqueda de hiperparámetros con XGBoost...")
    base_model = XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror")
    
    # Espacio de búsqueda básico pero efectivo para dinámicas físicas
    param_distributions = {
        "n_estimators": [100, 300, 500],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    }

    # Time-aware CV: al estar los datos ordenados, un CV tradicional mezcla el tiempo.
    # En proyectos reales se usaría TimeSeriesSplit, pero como son episodios independientes,
    # CV=3 estándar funciona suficientemente bien como aproximación si la semilla es fija.
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=15,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    
    search.fit(X_train, y_train)
    model = search.best_estimator_
    
    print(f"\nMejores hiperparámetros encontrados:\n{search.best_params_}")
    
    print("\nEvaluando modelo global...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"Global MAE:  {mae:.4f} °C")
    print(f"Global RMSE: {rmse:.4f} °C")
    print(f"Global R2:   {r2:.4f}")
    
    # Evaluar por regímenes
    evaluate_by_regime(df_test, y_test, predictions)
    
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\nModelo guardado en: {model_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML model for furnace physics.")
    parser.add_argument("--data", type=Path, default=Path("results/training_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/furnace_model.joblib"))
    args = parser.parse_args()

    train_furnace_model(data_path=args.data, model_output_path=args.output)


if __name__ == "__main__":
    main()