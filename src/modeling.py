import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, RANDOM_SEED, TARGET_COLUMN


def load_grouped_conditions(path):
    """Load the grouped fabrication-condition table used for model comparison."""
    data = pd.read_csv(path)
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required modeling column(s): {missing}")
    return data


def _make_model_definitions(n_samples):
    """Create model definitions appropriate for a small grouped dataset."""
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(len(FEATURE_COLUMNS)), length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1))
    )
    n_neighbors = min(3, max(1, n_samples - 1))

    models = [
        (
            "Gaussian Process Regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        GaussianProcessRegressor(
                            kernel=kernel,
                            normalize_y=True,
                            n_restarts_optimizer=5,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
        ),
        (
            "Random Forest Regressor",
            RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED),
        ),
        (
            "k-Nearest Neighbors Regressor",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsRegressor(n_neighbors=n_neighbors)),
                ]
            ),
        ),
    ]

    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("xgboost is not installed; skipping XGBoost Regressor.")
    else:
        models.append(
            (
                "XGBoost Regressor",
                XGBRegressor(
                    n_estimators=100,
                    max_depth=2,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=1.0,
                    objective="reg:squarederror",
                    random_state=RANDOM_SEED,
                ),
            )
        )

    return models


def _pearson_correlation(observed, predicted):
    """Calculate Pearson correlation, returning NaN when correlation is undefined."""
    if len(observed) < 2 or np.std(observed) == 0 or np.std(predicted) == 0:
        return np.nan
    return float(np.corrcoef(observed, predicted)[0, 1])


def _calculate_metrics(observed, predicted):
    """Calculate regression metrics over all Leave-One-Out predictions."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(observed, predicted))),
        "MAE": float(mean_absolute_error(observed, predicted)),
        "R2": float(r2_score(observed, predicted)),
        "Pearson_r": _pearson_correlation(observed, predicted),
    }


def compare_models_leave_one_out(grouped_conditions):
    """Compare regressors using Leave-One-Out CV on grouped fabrication conditions."""
    if len(grouped_conditions) < 3:
        raise ValueError("At least three grouped conditions are required for model comparison.")

    features = grouped_conditions[FEATURE_COLUMNS]
    target = grouped_conditions[TARGET_COLUMN]
    loo = LeaveOneOut()
    prediction_records = []
    metric_records = []

    for model_name, model in _make_model_definitions(len(grouped_conditions)):
        observed_values = []
        predicted_values = []

        for train_index, test_index in loo.split(features):
            x_train = features.iloc[train_index]
            y_train = target.iloc[train_index]
            x_test = features.iloc[test_index]
            y_test = target.iloc[test_index]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_train, y_train)

            prediction = float(model.predict(x_test)[0])
            observed = float(y_test.iloc[0])
            observed_values.append(observed)
            predicted_values.append(prediction)

            condition = grouped_conditions.iloc[test_index[0]]
            prediction_records.append(
                {
                    "SnO2_Rpm": condition["SnO2_Rpm"],
                    "Spiro_Oxid_Dur": condition["Spiro_Oxid_Dur"],
                    "observed_max_efficiency": observed,
                    "model_name": model_name,
                    "predicted_max_efficiency": prediction,
                    "prediction_error": prediction - observed,
                }
            )

        metrics = _calculate_metrics(np.array(observed_values), np.array(predicted_values))
        metric_records.append({"model_name": model_name, **metrics})

    metrics_data = pd.DataFrame(metric_records).sort_values("RMSE").reset_index(drop=True)
    predictions_data = pd.DataFrame(prediction_records)
    return metrics_data, predictions_data
