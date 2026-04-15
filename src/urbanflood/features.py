from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import ModelAssets, load_event_tables, list_event_ids


KEY_COLUMNS = ["model_id", "event_id", "node_type", "node_idx", "timestep"]


def _safe_divide(left: pd.Series, right: pd.Series) -> pd.Series:
    denom = right.replace(0, np.nan)
    return (left / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_edge_neighbor_features(
    node_dynamic: pd.DataFrame,
    edge_dynamic: pd.DataFrame,
    edge_index: pd.DataFrame,
    node_key: str,
    prefix: str,
) -> pd.DataFrame:
    # Convert edge-level flow and velocity into node-level summaries.
    merged = edge_dynamic.merge(edge_index, on="edge_idx", how="left")

    in_agg = (
        merged.groupby(["timestep", "to_node"])[["flow", "velocity"]]
        .agg(["mean", "max", "min"])
        .reset_index()
    )
    in_agg.columns = ["timestep", node_key] + [f"{prefix}_in_{a}_{b}" for a, b in in_agg.columns.tolist()[2:]]

    out_agg = (
        merged.groupby(["timestep", "from_node"])[["flow", "velocity"]]
        .agg(["mean", "max", "min"])
        .reset_index()
    )
    out_agg.columns = ["timestep", node_key] + [f"{prefix}_out_{a}_{b}" for a, b in out_agg.columns.tolist()[2:]]

    out = node_dynamic.merge(in_agg, on=["timestep", node_key], how="left")
    out = out.merge(out_agg, on=["timestep", node_key], how="left")
    return out


def _build_neighbor_water_level_features(
    node_dynamic: pd.DataFrame,
    edge_index: pd.DataFrame,
    node_key: str,
    prefix: str,
) -> pd.DataFrame:
    # Summarize upstream and downstream neighbor water levels for each node.
    node_levels = node_dynamic[["timestep", node_key, "water_level"]].copy()

    src = edge_index.rename(columns={"from_node": "neighbor_node", "to_node": node_key})
    src = src.merge(node_levels.rename(columns={node_key: "neighbor_node", "water_level": "neighbor_water_level"}),
                    on="neighbor_node", how="left")
    src_agg = (
        src.groupby(["timestep", node_key])["neighbor_water_level"]
        .agg(["mean", "max", "min"])
        .reset_index()
        .rename(columns={
            "mean": f"{prefix}_upstream_water_level_mean",
            "max": f"{prefix}_upstream_water_level_max",
            "min": f"{prefix}_upstream_water_level_min",
        })
    )

    dst = edge_index.rename(columns={"to_node": "neighbor_node", "from_node": node_key})
    dst = dst.merge(node_levels.rename(columns={node_key: "neighbor_node", "water_level": "neighbor_water_level"}),
                    on="neighbor_node", how="left")
    dst_agg = (
        dst.groupby(["timestep", node_key])["neighbor_water_level"]
        .agg(["mean", "max", "min"])
        .reset_index()
        .rename(columns={
            "mean": f"{prefix}_downstream_water_level_mean",
            "max": f"{prefix}_downstream_water_level_max",
            "min": f"{prefix}_downstream_water_level_min",
        })
    )

    out = node_dynamic.merge(src_agg, on=["timestep", node_key], how="left")
    out = out.merge(dst_agg, on=["timestep", node_key], how="left")
    return out


def _build_cross_domain_features(
    node_dynamic: pd.DataFrame,
    other_dynamic: pd.DataFrame,
    connections: pd.DataFrame,
    node_key: str,
    connection_other_key: str,
    other_dynamic_key: str,
    prefix: str,
) -> pd.DataFrame:
    # Join 1D-2D coupled node information so each node sees the paired domain.
    mapping = connections[[node_key, connection_other_key]].rename(columns={connection_other_key: "paired_node"})
    value_columns = [col for col in ["water_level", "rainfall", "water_volume"] if col in other_dynamic.columns]
    other_values = other_dynamic[["timestep", other_dynamic_key] + value_columns].rename(
        columns={other_dynamic_key: "paired_node"}
    )
    paired = mapping.merge(other_values, on="paired_node", how="left")

    agg_map = {col: ["mean", "max", "min"] for col in value_columns}
    paired_agg = paired.groupby(["timestep", node_key]).agg(agg_map).reset_index()
    paired_agg.columns = [
        "timestep",
        node_key,
        *[
            f"{prefix}_paired_{col}_{stat}"
            for col, stat in paired_agg.columns.tolist()[2:]
        ],
    ]
    return node_dynamic.merge(paired_agg, on=["timestep", node_key], how="left")


def _add_temporal_features(
    df: pd.DataFrame,
    static_level_col: str,
    lag_steps: tuple[int, ...],
    rain_windows: tuple[int, ...],
) -> pd.DataFrame:
    # Create the tabular time-series features used by the baseline models.
    df = df.sort_values(["model_id", "event_id", "node_type", "node_idx", "timestep"]).copy()
    group = df.groupby(["model_id", "event_id", "node_type", "node_idx"], sort=False)

    df["target_water_level"] = group["water_level"].shift(-1)
    df["target_delta_water_level"] = df["target_water_level"] - df["water_level"]
    df["delta_water_level"] = group["water_level"].diff().fillna(0.0)

    for lag in lag_steps:
        df[f"water_level_lag_{lag}"] = group["water_level"].shift(lag)
        df[f"delta_water_level_lag_{lag}"] = group["delta_water_level"].shift(lag)
        df[f"rainfall_lag_{lag}"] = group["rainfall"].shift(lag)

    for window in rain_windows:
        df[f"rainfall_sum_{window}"] = group["rainfall"].transform(
            lambda s: s.rolling(window, min_periods=1).sum()
        )
        df[f"rainfall_mean_{window}"] = group["rainfall"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

    df["cumulative_rainfall"] = group["rainfall"].cumsum()
    df["timestep_index"] = group.cumcount()
    df["water_level_above_reference"] = df["water_level"] - df[static_level_col]
    df["is_raining"] = (df["rainfall"] > 0).astype(np.int8)
    df["water_level_to_area_ratio"] = _safe_divide(df["water_level_above_reference"], df["effective_area"])

    return df


def build_node_frame(
    model_id: int,
    event_id: int,
    node_type: int,
    node_dynamic: pd.DataFrame,
    node_static: pd.DataFrame,
    edge_dynamic: pd.DataFrame,
    edge_index: pd.DataFrame,
    other_dynamic: pd.DataFrame,
    connections: pd.DataFrame,
    lag_steps: tuple[int, ...],
    rain_windows: tuple[int, ...],
) -> pd.DataFrame:
    # Build one node-centric panel for either 1D nodes or 2D nodes.
    node_key = "node_idx"
    prefix = "edge"

    frame = node_dynamic.copy()
    frame["model_id"] = model_id
    frame["event_id"] = event_id
    frame["node_type"] = node_type

    frame = frame.merge(node_static, on=node_key, how="left")
    frame = _build_edge_neighbor_features(frame, edge_dynamic, edge_index, node_key=node_key, prefix=prefix)
    frame = _build_neighbor_water_level_features(frame, edge_index, node_key=node_key, prefix=prefix)

    if node_type == 1:
        # 1D nodes do not have direct rainfall, so use rainfall from coupled 2D nodes.
        frame = _build_cross_domain_features(
            frame,
            other_dynamic,
            connections,
            node_key="node_idx",
            connection_other_key="paired_node_idx",
            other_dynamic_key="node_idx",
            prefix="surface",
        )
        frame["effective_area"] = frame["base_area"].fillna(0.0)
        frame["reference_elevation"] = frame["invert_elevation"].fillna(frame["surface_elevation"]).fillna(0.0)
        if "surface_paired_rainfall_mean" in frame.columns:
            frame["rainfall"] = frame["surface_paired_rainfall_mean"].fillna(0.0)
        else:
            frame["rainfall"] = 0.0
    else:
        # 2D nodes already observe rainfall directly in the dynamic table.
        frame = _build_cross_domain_features(
            frame,
            other_dynamic,
            connections,
            node_key="node_idx",
            connection_other_key="paired_node_idx",
            other_dynamic_key="node_idx",
            prefix="pipe",
        )
        frame["effective_area"] = frame["area"].fillna(0.0)
        frame["reference_elevation"] = frame["min_elevation"].fillna(frame["elevation"]).fillna(0.0)

    frame = _add_temporal_features(
        frame,
        static_level_col="reference_elevation",
        lag_steps=lag_steps,
        rain_windows=rain_windows,
    )

    return frame


def build_training_dataset(
    assets: ModelAssets,
    lag_steps: tuple[int, ...],
    rain_windows: tuple[int, ...],
    max_events: int | None = None,
) -> pd.DataFrame:
    # Concatenate every event into one training table with a unified schema.
    frames: list[pd.DataFrame] = []
    event_ids = list_event_ids(assets.split_dir)
    if max_events is not None:
        event_ids = event_ids[:max_events]

    for event_id in event_ids:
        tables = load_event_tables(assets.split_dir, event_id)

        frame_1d = build_node_frame(
            model_id=assets.model_id,
            event_id=event_id,
            node_type=1,
            node_dynamic=tables["1d_nodes_dynamic"].rename(columns={"inlet_flow": "node_aux_flow"}),
            node_static=assets.nodes_1d_static,
            edge_dynamic=tables["1d_edges_dynamic"],
            edge_index=assets.edges_1d_index,
            other_dynamic=tables["2d_nodes_dynamic"][["timestep", "node_idx", "water_level", "rainfall", "water_volume"]].copy(),
            connections=assets.connections_1d2d.rename(columns={"node_1d": "node_idx", "node_2d": "paired_node_idx"}),
            lag_steps=lag_steps,
            rain_windows=rain_windows,
        )

        frame_2d = build_node_frame(
            model_id=assets.model_id,
            event_id=event_id,
            node_type=2,
            node_dynamic=tables["2d_nodes_dynamic"],
            node_static=assets.nodes_2d_static,
            edge_dynamic=tables["2d_edges_dynamic"],
            edge_index=assets.edges_2d_index,
            other_dynamic=tables["1d_nodes_dynamic"][["timestep", "node_idx", "water_level"]].copy(),
            connections=assets.connections_1d2d.rename(columns={"node_2d": "node_idx", "node_1d": "paired_node_idx"}),
            lag_steps=lag_steps,
            rain_windows=rain_windows,
        )

        frames.extend([frame_1d, frame_2d])

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def clean_training_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Keep rows with a next-step target and fill remaining missing values safely.
    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["target_water_level"])
    object_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in object_cols:
        cleaned[col] = cleaned[col].fillna("")
    numeric_cols = cleaned.select_dtypes(include=[np.number, "bool"]).columns
    cleaned[numeric_cols] = cleaned[numeric_cols].fillna(0.0)
    return cleaned
