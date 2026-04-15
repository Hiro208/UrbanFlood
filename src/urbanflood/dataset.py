from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .io import read_csv


@dataclass(slots=True)
class ModelAssets:
    # Static tables and graph connectivity shared by all events of one model.
    model_id: int
    split_dir: Path
    nodes_1d_static: pd.DataFrame
    nodes_2d_static: pd.DataFrame
    edges_1d_static: pd.DataFrame
    edges_2d_static: pd.DataFrame
    edges_1d_index: pd.DataFrame
    edges_2d_index: pd.DataFrame
    connections_1d2d: pd.DataFrame


def list_event_ids(split_dir: Path) -> list[int]:
    # Event folders are stored as event_{id}.
    event_ids = []
    for child in split_dir.iterdir():
        if child.is_dir() and child.name.startswith("event_"):
            event_ids.append(int(child.name.split("_")[1]))
    return sorted(event_ids)


def load_model_assets(data_dir: Path, model_id: int, split: str = "train") -> ModelAssets:
    # Load model-level static data once so event processing can reuse it.
    split_dir = data_dir / f"Model_{model_id}" / split
    return ModelAssets(
        model_id=model_id,
        split_dir=split_dir,
        nodes_1d_static=read_csv(split_dir / "1d_nodes_static.csv"),
        nodes_2d_static=read_csv(split_dir / "2d_nodes_static.csv"),
        edges_1d_static=read_csv(split_dir / "1d_edges_static.csv"),
        edges_2d_static=read_csv(split_dir / "2d_edges_static.csv"),
        edges_1d_index=read_csv(split_dir / "1d_edge_index.csv"),
        edges_2d_index=read_csv(split_dir / "2d_edge_index.csv"),
        connections_1d2d=read_csv(split_dir / "1d2d_connections.csv"),
    )


def load_event_tables(split_dir: Path, event_id: int) -> dict[str, pd.DataFrame]:
    # Load the dynamic files for a single rainfall event.
    event_dir = split_dir / f"event_{event_id}"
    return {
        "timesteps": read_csv(event_dir / "timesteps.csv"),
        "1d_nodes_dynamic": read_csv(event_dir / "1d_nodes_dynamic_all.csv"),
        "2d_nodes_dynamic": read_csv(event_dir / "2d_nodes_dynamic_all.csv"),
        "1d_edges_dynamic": read_csv(event_dir / "1d_edges_dynamic_all.csv"),
        "2d_edges_dynamic": read_csv(event_dir / "2d_edges_dynamic_all.csv"),
    }
