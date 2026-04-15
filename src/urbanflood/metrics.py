from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, part in df.groupby("model_name", sort=False):
        rows.append(
            {
                "model_name": model_name,
                "rmse": rmse(part["target_water_level"], part["prediction"]),
                "mae": mae(part["target_water_level"], part["prediction"]),
                "num_rows": int(len(part)),
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
