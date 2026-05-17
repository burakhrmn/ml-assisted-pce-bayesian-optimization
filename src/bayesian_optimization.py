import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    EXPECTED_IMPROVEMENT_XI,
    FEATURE_COLUMNS,
    MIN_EXISTING_RPM_SEPARATION,
    MIN_EXISTING_SPIRO_SEPARATION,
    MIN_RECOMMENDATION_RPM_SEPARATION,
    MIN_RECOMMENDATION_SPIRO_SEPARATION,
    N_RECOMMENDATIONS,
    RANDOM_SEED,
    SNO2_RPM_BOUNDS,
    SNO2_RPM_GRID_STEP,
    SPIRO_OXID_DUR_BOUNDS,
    SPIRO_OXID_DUR_GRID_STEP,
    TARGET_COLUMN,
)


def fit_gpr_surrogate(grouped_conditions):
    """Fit a Gaussian Process surrogate on grouped fabrication conditions."""
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(len(FEATURE_COLUMNS)), length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1))
    )
    surrogate = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    n_restarts_optimizer=10,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surrogate.fit(grouped_conditions[FEATURE_COLUMNS], grouped_conditions[TARGET_COLUMN])

    return surrogate


def generate_candidate_grid():
    """Generate a dense candidate grid inside the experimentally explored bounds."""
    rpm_values = np.arange(
        SNO2_RPM_BOUNDS[0],
        SNO2_RPM_BOUNDS[1] + SNO2_RPM_GRID_STEP,
        SNO2_RPM_GRID_STEP,
    )
    spiro_values = np.arange(
        SPIRO_OXID_DUR_BOUNDS[0],
        SPIRO_OXID_DUR_BOUNDS[1] + SPIRO_OXID_DUR_GRID_STEP,
        SPIRO_OXID_DUR_GRID_STEP,
    )
    grid = np.array(np.meshgrid(rpm_values, spiro_values)).T.reshape(-1, 2)
    return pd.DataFrame(grid, columns=FEATURE_COLUMNS)


def _predict_with_uncertainty(surrogate, candidates):
    """Predict candidate mean and standard deviation from a fitted pipeline."""
    scaler = surrogate.named_steps["scaler"]
    model = surrogate.named_steps["model"]
    scaled_candidates = scaler.transform(candidates[FEATURE_COLUMNS])
    mean, std = model.predict(scaled_candidates, return_std=True)
    return mean, std


def _expected_improvement(mean, std, best_observed, xi=EXPECTED_IMPROVEMENT_XI):
    """Calculate Expected Improvement for maximization."""
    std = np.maximum(std, 1e-12)
    improvement = mean - best_observed - xi
    z_score = improvement / std
    expected_improvement = improvement * norm.cdf(z_score) + std * norm.pdf(z_score)
    expected_improvement[std <= 1e-12] = 0.0
    return expected_improvement


def _add_existing_condition_metadata(candidates, grouped_conditions):
    """Mark tested points and attach nearest tested-condition information."""
    existing = grouped_conditions[FEATURE_COLUMNS].copy()
    candidate_points = candidates[FEATURE_COLUMNS].to_numpy(dtype=float)
    existing_points = existing.to_numpy(dtype=float)

    ranges = np.array(
        [
            SNO2_RPM_BOUNDS[1] - SNO2_RPM_BOUNDS[0],
            SPIRO_OXID_DUR_BOUNDS[1] - SPIRO_OXID_DUR_BOUNDS[0],
        ],
        dtype=float,
    )
    normalized_distances = cdist(candidate_points / ranges, existing_points / ranges)
    nearest_indices = normalized_distances.argmin(axis=1)
    nearest_distances = normalized_distances.min(axis=1)
    nearest_points = existing_points[nearest_indices]

    existing_tuples = set(map(tuple, existing_points))
    candidates["is_existing_condition"] = [
        tuple(point) in existing_tuples for point in candidate_points
    ]
    candidates["nearest_existing_condition"] = [
        (
            f"SnO2_Rpm={existing_points[index][0]:.0f}, "
            f"Spiro_Oxid_Dur={existing_points[index][1]:.2f}"
        )
        for index in nearest_indices
    ]
    candidates["nearest_existing_SnO2_Rpm"] = nearest_points[:, 0]
    candidates["nearest_existing_Spiro_Oxid_Dur"] = nearest_points[:, 1]
    candidates["delta_SnO2_Rpm"] = candidates["SnO2_Rpm"] - candidates["nearest_existing_SnO2_Rpm"]
    candidates["delta_Spiro_Oxid_Dur"] = (
        candidates["Spiro_Oxid_Dur"] - candidates["nearest_existing_Spiro_Oxid_Dur"]
    )
    candidates["distance_to_nearest_existing_condition"] = nearest_distances
    return candidates


def _select_diverse_recommendations(ranked_candidates, n_recommendations=N_RECOMMENDATIONS):
    """Select ranked candidates with minimum physical separation between picks."""
    selected = []
    for _, candidate in ranked_candidates.iterrows():
        if candidate["is_existing_condition"]:
            continue
        if _is_too_close_to_existing_condition(candidate):
            continue
        if _is_sufficiently_separated(candidate, selected):
            selected.append(candidate)
        if len(selected) == n_recommendations:
            break

    recommendations = pd.DataFrame(selected).reset_index(drop=True)
    recommendations.insert(0, "rank", recommendations.index + 1)
    return recommendations


def _is_too_close_to_existing_condition(candidate):
    """Check whether a candidate is nearly identical to its nearest tested condition."""
    return (
        abs(candidate["delta_SnO2_Rpm"]) < MIN_EXISTING_RPM_SEPARATION
        and abs(candidate["delta_Spiro_Oxid_Dur"]) < MIN_EXISTING_SPIRO_SEPARATION
    )


def _is_sufficiently_separated(candidate, selected):
    """Check whether a candidate is physically distinct from selected recommendations."""
    for existing_candidate in selected:
        rpm_delta = abs(candidate["SnO2_Rpm"] - existing_candidate["SnO2_Rpm"])
        spiro_delta = abs(candidate["Spiro_Oxid_Dur"] - existing_candidate["Spiro_Oxid_Dur"])
        if (
            rpm_delta < MIN_RECOMMENDATION_RPM_SEPARATION
            or spiro_delta < MIN_RECOMMENDATION_SPIRO_SEPARATION
        ):
            return False
    return True


def _format_recommendations(recommendations):
    """Return recommendation columns in an experiment-facing order."""
    return recommendations[
        [
            "rank",
            "SnO2_Rpm",
            "Spiro_Oxid_Dur",
            "predicted_max_efficiency",
            "uncertainty",
            "expected_improvement",
            "ucb_beta_1_96",
            "nearest_existing_condition",
            "nearest_existing_SnO2_Rpm",
            "nearest_existing_Spiro_Oxid_Dur",
            "delta_SnO2_Rpm",
            "delta_Spiro_Oxid_Dur",
            "distance_to_nearest_existing_condition",
        ]
    ]


def _rank_diverse(candidates, sort_columns, ascending):
    """Rank candidates and return physically diverse recommendations."""
    ranked = candidates.sort_values(sort_columns, ascending=ascending)
    return _format_recommendations(_select_diverse_recommendations(ranked))


def _fit_exploitation_models(grouped_conditions):
    """Fit candidate-screening models on grouped fabrication conditions."""
    x_data = grouped_conditions[FEATURE_COLUMNS]
    y_data = grouped_conditions[TARGET_COLUMN]
    n_neighbors = min(3, max(1, len(grouped_conditions) - 1))
    models = {
        "gpr_predicted": fit_gpr_surrogate(grouped_conditions),
        "rf_predicted": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_SEED,
        ),
        "knn_predicted": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=n_neighbors)),
            ]
        ),
    }

    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("xgboost is not installed; skipping XGBoost exploitation predictions.")
    else:
        models["xgb_predicted"] = XGBRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=1.0,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
        )

    fitted_models = {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_data, y_data)
        fitted_models[name] = model
    return fitted_models


def build_model_candidate_predictions(grouped_conditions, candidates, best_observed):
    """Predict candidate performance with multiple models and ensemble summaries."""
    model_predictions = candidates[["SnO2_Rpm", "Spiro_Oxid_Dur"]].copy()
    fitted_models = _fit_exploitation_models(grouped_conditions)

    for prediction_column, model in fitted_models.items():
        model_predictions[prediction_column] = model.predict(candidates[FEATURE_COLUMNS])

    prediction_columns = [
        column for column in model_predictions.columns if column.endswith("_predicted")
    ]
    model_predictions["ensemble_mean_prediction"] = model_predictions[prediction_columns].mean(axis=1)
    model_predictions["ensemble_std_prediction"] = model_predictions[prediction_columns].std(
        axis=1,
        ddof=0,
    )
    model_predictions["ensemble_upper_score"] = (
        model_predictions["ensemble_mean_prediction"]
        + model_predictions["ensemble_std_prediction"]
    )
    model_predictions["exceeds_best_observed_ensemble_mean"] = (
        model_predictions["ensemble_mean_prediction"] > best_observed
    )
    model_predictions["exceeds_best_observed_ensemble_upper"] = (
        model_predictions["ensemble_upper_score"] > best_observed
    )

    return model_predictions


def build_ensemble_recommendations(candidates, model_predictions):
    """Select diverse recommendations ranked by ensemble upper score."""
    ranked = candidates.merge(
        model_predictions[
            [
                "SnO2_Rpm",
                "Spiro_Oxid_Dur",
                "ensemble_mean_prediction",
                "ensemble_upper_score",
            ]
        ],
        on=["SnO2_Rpm", "Spiro_Oxid_Dur"],
        how="left",
    ).sort_values(
        ["ensemble_upper_score", "ensemble_mean_prediction", "expected_improvement"],
        ascending=[False, False, False],
    )
    recommendations = _select_diverse_recommendations(ranked)
    recommendations = _format_recommendations(recommendations)
    return recommendations.merge(
        model_predictions[
            [
                "SnO2_Rpm",
                "Spiro_Oxid_Dur",
                "ensemble_mean_prediction",
                "ensemble_upper_score",
            ]
        ],
        on=["SnO2_Rpm", "Spiro_Oxid_Dur"],
        how="left",
    )


def build_champion_challenger_table(
    ei_recommendations,
    ucb_recommendations,
    predicted_recommendations,
    ensemble_recommendations,
):
    """Combine top recommendation strategies into a deduplicated experiment table."""
    sources = [
        ("EI-diverse", ei_recommendations, "High Expected Improvement from GPR surrogate."),
        ("UCB-diverse", ucb_recommendations, "High GPR upper confidence bound."),
        (
            "Predicted-efficiency-diverse",
            predicted_recommendations,
            "High GPR predicted max efficiency under diversity constraints.",
        ),
        (
            "Ensemble-upper-diverse",
            ensemble_recommendations,
            "High ensemble upper score across candidate-screening models.",
        ),
    ]
    records = []
    seen = set()
    for source, table, rationale in sources:
        for row in table.head(5).itertuples(index=False):
            key = (row.SnO2_Rpm, row.Spiro_Oxid_Dur)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "recommendation_source": source,
                    "SnO2_Rpm": row.SnO2_Rpm,
                    "Spiro_Oxid_Dur": row.Spiro_Oxid_Dur,
                    "predicted_max_efficiency": row.predicted_max_efficiency,
                    "uncertainty": row.uncertainty,
                    "expected_improvement": row.expected_improvement,
                    "ucb_beta_1_96": row.ucb_beta_1_96,
                    "ensemble_mean_prediction": getattr(row, "ensemble_mean_prediction", np.nan),
                    "ensemble_upper_score": getattr(row, "ensemble_upper_score", np.nan),
                    "rationale": rationale,
                }
            )

    return pd.DataFrame(records)


def build_gpr_diagnostics(grouped_conditions, candidates, surrogate):
    """Build a concise diagnostics report for the fitted GPR and BO grid."""
    best_condition = grouped_conditions.sort_values(TARGET_COLUMN, ascending=False).iloc[0]
    model = surrogate.named_steps["model"]
    exact_existing_count = int(candidates["is_existing_condition"].sum())
    top_ei = candidates.sort_values("expected_improvement", ascending=False).head(N_RECOMMENDATIONS)
    top_ei_range = top_ei["expected_improvement"].max() - top_ei["expected_improvement"].min()
    near_identical_note = (
        "Many top candidates have nearly identical EI values."
        if top_ei_range < 1e-4
        else "Top candidate EI values show measurable spread."
    )

    lines = [
        "Gaussian Process Regression diagnostics",
        f"Optimized kernel: {model.kernel_}",
        (
            "Best observed condition: "
            f"SnO2_Rpm={best_condition['SnO2_Rpm']}, "
            f"Spiro_Oxid_Dur={best_condition['Spiro_Oxid_Dur']}"
        ),
        f"Best observed max_efficiency: {best_condition[TARGET_COLUMN]:.6f}",
        f"Number of candidate grid points: {len(candidates)}",
        f"Number of exact existing conditions excluded: {exact_existing_count}",
        (
            "Candidates with GPR mean above best observed: "
            f"{int(candidates['exceeds_best_observed_prediction'].sum())}"
        ),
        (
            "Candidates with GPR UCB beta=1.96 above best observed: "
            f"{int(candidates['exceeds_best_observed_ucb_1_96'].sum())}"
        ),
        f"Top-{N_RECOMMENDATIONS} EI range: {top_ei_range:.8f}",
        f"EI diagnostic note: {near_identical_note}",
    ]
    return "\n".join(lines) + "\n"


def run_bayesian_optimization(grouped_conditions):
    """Fit the GPR surrogate and rank diverse new experimental candidates."""
    surrogate = fit_gpr_surrogate(grouped_conditions)
    candidates = generate_candidate_grid()
    mean, std = _predict_with_uncertainty(surrogate, candidates)
    best_observed = grouped_conditions[TARGET_COLUMN].max()

    candidates["predicted_max_efficiency"] = mean
    candidates["uncertainty"] = std
    candidates["expected_improvement"] = _expected_improvement(mean, std, best_observed)
    candidates["ucb_beta_1"] = candidates["predicted_max_efficiency"] + candidates["uncertainty"]
    candidates["ucb_beta_1_96"] = (
        candidates["predicted_max_efficiency"] + 1.96 * candidates["uncertainty"]
    )
    candidates["ucb_beta_2"] = (
        candidates["predicted_max_efficiency"] + 2.0 * candidates["uncertainty"]
    )
    candidates = _add_existing_condition_metadata(candidates, grouped_conditions)
    candidates["exceeds_best_observed_prediction"] = (
        candidates["predicted_max_efficiency"] > best_observed
    )
    candidates["exceeds_best_observed_ucb_1_96"] = (
        candidates["ucb_beta_1_96"] > best_observed
    )

    ordered_columns = [
        "SnO2_Rpm",
        "Spiro_Oxid_Dur",
        "predicted_max_efficiency",
        "uncertainty",
        "expected_improvement",
        "ucb_beta_1",
        "ucb_beta_1_96",
        "ucb_beta_2",
        "is_existing_condition",
        "nearest_existing_condition",
        "nearest_existing_SnO2_Rpm",
        "nearest_existing_Spiro_Oxid_Dur",
        "delta_SnO2_Rpm",
        "delta_Spiro_Oxid_Dur",
        "distance_to_nearest_existing_condition",
        "exceeds_best_observed_prediction",
        "exceeds_best_observed_ucb_1_96",
    ]
    candidates = candidates[ordered_columns]

    ei_recommendations = _rank_diverse(
        candidates,
        ["expected_improvement", "predicted_max_efficiency", "uncertainty"],
        [False, False, False],
    )
    ucb_recommendations = _rank_diverse(
        candidates,
        ["ucb_beta_1_96", "expected_improvement", "predicted_max_efficiency"],
        [False, False, False],
    )
    predicted_recommendations = _rank_diverse(
        candidates,
        ["predicted_max_efficiency", "expected_improvement", "uncertainty"],
        [False, False, False],
    )
    model_predictions = build_model_candidate_predictions(
        grouped_conditions,
        candidates,
        best_observed,
    )
    ensemble_recommendations = build_ensemble_recommendations(candidates, model_predictions)
    champion_challenger = build_champion_challenger_table(
        ei_recommendations,
        ucb_recommendations,
        predicted_recommendations,
        ensemble_recommendations,
    )
    diagnostics = build_gpr_diagnostics(grouped_conditions, candidates, surrogate)

    return (
        candidates.sort_values(
            ["expected_improvement", "predicted_max_efficiency", "uncertainty"],
            ascending=[False, False, False],
        ).reset_index(drop=True),
        ei_recommendations,
        predicted_recommendations,
        ucb_recommendations,
        model_predictions,
        ensemble_recommendations,
        champion_challenger,
        diagnostics,
        surrogate,
    )
