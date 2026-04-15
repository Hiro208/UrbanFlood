from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dataset import list_event_ids
from .io import save_frame


def _build_test_predictions_for_node_type(
    model_id: int,
    event_id: int,
    node_type: int,
    dynamic_path: Path,
) -> pd.DataFrame:
    # Use the last observed test value for each node to fill all future rows.
    df = pd.read_csv(dynamic_path)
    future_rows = df[df["water_level"].isna()].copy()
    observed_rows = df[df["water_level"].notna()].copy()

    if future_rows.empty:
        return pd.DataFrame(
            columns=["model_id", "event_id", "node_type", "timestep", "node_id", "water_level"]
        )

    last_known = (
        observed_rows.sort_values(["node_idx", "timestep"])
        .groupby("node_idx", as_index=False)
        .tail(1)[["node_idx", "water_level"]]
        .rename(columns={"water_level": "predicted_water_level"})
    )

    out = future_rows[["timestep", "node_idx"]].merge(last_known, on="node_idx", how="left")
    out["model_id"] = model_id
    out["event_id"] = event_id
    out["node_type"] = node_type
    out["node_id"] = out["node_idx"]
    out["water_level"] = out["predicted_water_level"]

    return out[["model_id", "event_id", "node_type", "timestep", "node_id", "water_level"]]


def build_persistence_submission(data_dir: Path) -> pd.DataFrame:
    # Build a Kaggle-ready submission across both models and all test events.
    parts: list[pd.DataFrame] = []

    for model_id in (1, 2):
        split_dir = data_dir / f"Model_{model_id}" / "test"
        event_ids = list_event_ids(split_dir)

        for event_id in event_ids:
            event_dir = split_dir / f"event_{event_id}"
            part_1d = _build_test_predictions_for_node_type(
                model_id=model_id,
                event_id=event_id,
                node_type=1,
                dynamic_path=event_dir / "1d_nodes_dynamic_all.csv",
            )
            part_2d = _build_test_predictions_for_node_type(
                model_id=model_id,
                event_id=event_id,
                node_type=2,
                dynamic_path=event_dir / "2d_nodes_dynamic_all.csv",
            )
            parts.extend([part_1d, part_2d])

    submission = pd.concat(parts, ignore_index=True)
    submission = submission.sort_values(
        ["model_id", "event_id", "node_type", "timestep", "node_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    submission.insert(0, "row_id", range(len(submission)))
    return submission[["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"]]


def save_persistence_submission(data_dir: Path, output_path: Path) -> pd.DataFrame:
    # Helper used by the notebook to export the final CSV in one call.
    submission = build_persistence_submission(data_dir)
    save_frame(submission, output_path)
    return submission
