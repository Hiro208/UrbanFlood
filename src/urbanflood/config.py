from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class BuildConfig:
    data_dir: Path
    output_dir: Path
    lag_steps: tuple[int, ...] = (1, 2, 3, 5, 10)
    rain_windows: tuple[int, ...] = (3, 5, 10)
    train_split: str = "train"
    models: tuple[int, ...] = (1, 2)
    save_parquet: bool = True
    max_events_per_model: int | None = None


@dataclass(slots=True)
class BaselineConfig:
    data_dir: Path
    output_dir: Path
    lag_steps: tuple[int, ...] = (1, 2, 3, 5, 10)
    rain_windows: tuple[int, ...] = (3, 5, 10)
    validation_fraction: float = 0.2
    random_state: int = 42
    train_split: str = "train"
    models: tuple[int, ...] = (1, 2)
    n_estimators: int = 300
    max_depth: int | None = 18
    min_samples_leaf: int = 2
    n_jobs: int = -1
    max_events_per_model: int | None = None
    max_train_rows: int | None = 300_000
    drop_columns: tuple[str, ...] = field(
        default_factory=lambda: (
            "target_water_level",
            "target_delta_water_level",
            "timestamp",
        )
    )
