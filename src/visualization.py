import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import EFFICIENCY_COLUMN, EFFICIENCY_THRESHOLD, FEATURE_COLUMNS


DPI = 300
FIGURE_SIZE = (7, 5)


def _prepare_figure_dir(figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)


def _save_current_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_efficiency_threshold_histogram(raw_data, filtered_data, figure_dir):
    _prepare_figure_dir(figure_dir)

    plt.figure(figsize=FIGURE_SIZE)
    plt.hist(
        raw_data[EFFICIENCY_COLUMN],
        bins=30,
        alpha=0.55,
        label="Before filtering",
        color="#4C78A8",
        edgecolor="white",
    )
    plt.hist(
        filtered_data[EFFICIENCY_COLUMN],
        bins=30,
        alpha=0.55,
        label="After filtering",
        color="#F58518",
        edgecolor="white",
    )
    plt.axvline(
        EFFICIENCY_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=f"Threshold = {EFFICIENCY_THRESHOLD}%",
    )
    plt.xlabel("Efficiency (%)")
    plt.ylabel("Count")
    plt.legend(frameon=False)
    _save_current_figure(figure_dir / "efficiency_threshold_histogram.png")


def _plot_grouped_heatmap(grouped_conditions, value_column, output_name, figure_dir, fmt=".2f"):
    _prepare_figure_dir(figure_dir)

    heatmap_data = grouped_conditions.pivot(
        index=FEATURE_COLUMNS[0],
        columns=FEATURE_COLUMNS[1],
        values=value_column,
    ).sort_index(ascending=True)

    plt.figure(figsize=FIGURE_SIZE)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=fmt,
        cmap="viridis",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": value_column.replace("_", " ").title()},
    )
    plt.xlabel("Spiro_Oxid_Dur")
    plt.ylabel("SnO2_Rpm")
    _save_current_figure(figure_dir / output_name)


def plot_existing_experimental_conditions_map(grouped_conditions, figure_dir):
    _prepare_figure_dir(figure_dir)

    plt.figure(figsize=FIGURE_SIZE)
    scatter = plt.scatter(
        grouped_conditions["Spiro_Oxid_Dur"],
        grouped_conditions["SnO2_Rpm"],
        s=grouped_conditions["count"] * 12,
        c=grouped_conditions["max_efficiency"],
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
    )
    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Max Efficiency (%)")
    plt.xlabel("Spiro_Oxid_Dur")
    plt.ylabel("SnO2_Rpm")
    _save_current_figure(figure_dir / "existing_experimental_conditions_map.png")


def save_all_figures(raw_data, filtered_data, grouped_conditions, figure_dir):
    plot_efficiency_threshold_histogram(raw_data, filtered_data, figure_dir)
    _plot_grouped_heatmap(
        grouped_conditions,
        "mean_efficiency",
        "mean_efficiency_heatmap.png",
        figure_dir,
    )
    _plot_grouped_heatmap(
        grouped_conditions,
        "max_efficiency",
        "max_efficiency_heatmap.png",
        figure_dir,
    )
    _plot_grouped_heatmap(
        grouped_conditions,
        "std_efficiency",
        "std_efficiency_heatmap.png",
        figure_dir,
    )
    _plot_grouped_heatmap(
        grouped_conditions,
        "count",
        "count_heatmap.png",
        figure_dir,
        fmt=".0f",
    )
    plot_existing_experimental_conditions_map(grouped_conditions, figure_dir)


def plot_model_comparison_rmse(metrics_data, figure_dir):
    """Save a bar plot comparing model RMSE values."""
    _prepare_figure_dir(figure_dir)

    ordered = metrics_data.sort_values("RMSE")
    plt.figure(figsize=FIGURE_SIZE)
    sns.barplot(data=ordered, x="RMSE", y="model_name", color="#4C78A8")
    plt.xlabel("RMSE")
    plt.ylabel("")
    _save_current_figure(figure_dir / "model_comparison_rmse.png")


def plot_model_comparison_r2(metrics_data, figure_dir):
    """Save a bar plot comparing model R-squared values."""
    _prepare_figure_dir(figure_dir)

    ordered = metrics_data.sort_values("R2", ascending=False)
    plt.figure(figsize=FIGURE_SIZE)
    sns.barplot(data=ordered, x="R2", y="model_name", color="#54A24B")
    plt.axvline(0, color="black", linewidth=0.9)
    plt.xlabel("R2")
    plt.ylabel("")
    _save_current_figure(figure_dir / "model_comparison_r2.png")


def plot_predicted_vs_observed_models(predictions_data, figure_dir):
    """Save observed-vs-predicted scatter plot for all compared models."""
    _prepare_figure_dir(figure_dir)

    observed = predictions_data["observed_max_efficiency"]
    predicted = predictions_data["predicted_max_efficiency"]
    axis_min = min(observed.min(), predicted.min()) - 0.5
    axis_max = max(observed.max(), predicted.max()) + 0.5

    plt.figure(figsize=FIGURE_SIZE)
    sns.scatterplot(
        data=predictions_data,
        x="observed_max_efficiency",
        y="predicted_max_efficiency",
        hue="model_name",
        style="model_name",
        s=70,
        edgecolor="black",
        linewidth=0.3,
    )
    plt.plot([axis_min, axis_max], [axis_min, axis_max], color="black", linestyle="--")
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    plt.xlabel("Observed max efficiency (%)")
    plt.ylabel("Predicted max efficiency (%)")
    plt.legend(frameon=False)
    _save_current_figure(figure_dir / "predicted_vs_observed_models.png")


def save_model_comparison_figures(metrics_data, predictions_data, figure_dir):
    """Save all model-comparison figures."""
    plot_model_comparison_rmse(metrics_data, figure_dir)
    plot_model_comparison_r2(metrics_data, figure_dir)
    plot_predicted_vs_observed_models(predictions_data, figure_dir)


def _plot_candidate_surface(candidates, value_column, output_name, colorbar_label, figure_dir):
    """Save a heatmap-like surface from dense BO candidate predictions."""
    _prepare_figure_dir(figure_dir)

    surface = candidates.pivot(
        index="SnO2_Rpm",
        columns="Spiro_Oxid_Dur",
        values=value_column,
    ).sort_index(ascending=True)

    plt.figure(figsize=(8, 5.6))
    mesh = plt.pcolormesh(
        surface.columns,
        surface.index,
        surface.values,
        shading="auto",
        cmap="viridis",
    )
    colorbar = plt.colorbar(mesh)
    colorbar.set_label(colorbar_label)
    plt.xlabel("Spiro-OMeTAD oxidation duration (h)")
    plt.ylabel("SnO₂ spin-coating speed (rpm)")
    _save_current_figure(figure_dir / output_name)


def plot_recommended_experiments_map(
    grouped_conditions,
    ei_recommendations,
    figure_dir,
    ucb_recommendations=None,
    ensemble_recommendations=None,
):
    """Save a map comparing existing and recommended fabrication conditions."""
    _prepare_figure_dir(figure_dir)

    plt.figure(figsize=(8, 5.6))
    plt.scatter(
        grouped_conditions["Spiro_Oxid_Dur"],
        grouped_conditions["SnO2_Rpm"],
        marker="o",
        s=55,
        facecolors="none",
        edgecolors="#4C78A8",
        linewidths=1.2,
        label="Existing conditions",
    )
    plt.scatter(
        ei_recommendations["Spiro_Oxid_Dur"],
        ei_recommendations["SnO2_Rpm"],
        marker="*",
        s=130,
        color="#E45756",
        edgecolors="black",
        linewidths=0.4,
        label="EI recommendations",
    )
    if ucb_recommendations is not None:
        plt.scatter(
            ucb_recommendations["Spiro_Oxid_Dur"],
            ucb_recommendations["SnO2_Rpm"],
            marker="^",
            s=90,
            color="#F58518",
            edgecolors="black",
            linewidths=0.4,
            label="UCB recommendations",
        )
    if ensemble_recommendations is not None:
        plt.scatter(
            ensemble_recommendations["Spiro_Oxid_Dur"],
            ensemble_recommendations["SnO2_Rpm"],
            marker="s",
            s=70,
            color="#54A24B",
            edgecolors="black",
            linewidths=0.4,
            label="Ensemble recommendations",
        )
    plt.xlabel("Spiro-OMeTAD oxidation duration (h)")
    plt.ylabel("SnO₂ spin-coating speed (rpm)")
    plt.legend(frameon=False)
    _save_current_figure(figure_dir / "recommended_experiments_map.png")


def save_bayesian_optimization_figures(
    candidates,
    grouped_conditions,
    ei_recommendations,
    figure_dir,
    ucb_recommendations=None,
    model_predictions=None,
    ensemble_recommendations=None,
):
    """Save all Bayesian Optimization response and acquisition figures."""
    _plot_candidate_surface(
        candidates,
        "predicted_max_efficiency",
        "predicted_response_surface.png",
        "Predicted max efficiency (%)",
        figure_dir,
    )
    _plot_candidate_surface(
        candidates,
        "uncertainty",
        "uncertainty_surface.png",
        "Predictive uncertainty",
        figure_dir,
    )
    _plot_candidate_surface(
        candidates,
        "expected_improvement",
        "expected_improvement_map.png",
        "Expected Improvement",
        figure_dir,
    )
    _plot_candidate_surface(
        candidates,
        "ucb_beta_1_96",
        "ucb_beta_1_96_map.png",
        "UCB beta=1.96",
        figure_dir,
    )
    if model_predictions is not None and "ensemble_upper_score" in model_predictions.columns:
        _plot_candidate_surface(
            model_predictions,
            "ensemble_upper_score",
            "ensemble_upper_score_map.png",
            "Ensemble upper score",
            figure_dir,
        )
    plot_recommended_experiments_map(
        grouped_conditions,
        ei_recommendations,
        figure_dir,
        ucb_recommendations=ucb_recommendations,
        ensemble_recommendations=ensemble_recommendations,
    )
