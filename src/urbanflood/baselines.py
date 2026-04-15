from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .metrics import summarize_metrics


ID_COLUMNS = ["model_id", "event_id", "node_type", "node_idx", "timestep"]


def event_based_split(
    df: pd.DataFrame,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Split by event instead of random rows to avoid timestep leakage.
    val_parts = []
    train_parts = []

    for (model_id, node_type), part in df.groupby(["model_id", "node_type"], sort=False):
        event_ids = sorted(part["event_id"].unique().tolist())
        n_val = max(1, round(len(event_ids) * validation_fraction))
        val_event_ids = set(event_ids[-n_val:])
        train_parts.append(part[~part["event_id"].isin(val_event_ids)].copy())
        val_parts.append(part[part["event_id"].isin(val_event_ids)].copy())

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    return train_df, val_df


def get_feature_columns(df: pd.DataFrame, drop_columns: tuple[str, ...]) -> list[str]:
    excluded = set(ID_COLUMNS) | set(drop_columns)
    return [col for col in df.columns if col not in excluded]


def run_persistence_baseline(val_df: pd.DataFrame) -> pd.DataFrame:
    # Predict the next step as the current observed water level.
    out = val_df[ID_COLUMNS + ["target_water_level"]].copy()
    out["prediction"] = val_df["water_level"]
    out["model_name"] = "persistence"
    return out


def run_random_forest_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    random_state: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, RandomForestRegressor]:
    # Train a simple non-neural baseline on the engineered node-level features.
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(train_df[feature_columns], train_df["target_water_level"])

    out = val_df[ID_COLUMNS + ["target_water_level"]].copy()
    out["prediction"] = model.predict(val_df[feature_columns])
    out["model_name"] = "random_forest"
    return out, model


def evaluate_baselines(predictions: list[pd.DataFrame]) -> pd.DataFrame:
    # Compare all baseline predictions in one summary table.
    all_predictions = pd.concat(predictions, ignore_index=True)
    return summarize_metrics(all_predictions)
