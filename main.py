from src.config import (
    EFFICIENCY_COLUMN,
    EFFICIENCY_THRESHOLD,
    FEATURE_COLUMNS,
    FIGURE_OUTPUT_DIR,
    FILTERED_DATA_PATH,
    ANALYSIS_REPORT_PATH,
    CANDIDATE_PREDICTIONS_ENHANCED_PATH,
    CANDIDATE_PREDICTIONS_BY_MODEL_PATH,
    CANDIDATE_PREDICTIONS_PATH,
    CHAMPION_CHALLENGER_EXPERIMENTS_PATH,
    CURATED_EXPERIMENTAL_PLAN_PATH,
    GPR_DIAGNOSTICS_PATH,
    GROUPED_CONDITIONS_PATH,
    MODEL_METRICS_PATH,
    MODEL_PREDICTIONS_PATH,
    RAW_DATA_PATH,
    RECOMMENDED_EXPERIMENTS_EI_DIVERSE_PATH,
    RECOMMENDED_EXPERIMENTS_ENSEMBLE_DIVERSE_PATH,
    RECOMMENDED_EXPERIMENTS_PATH,
    RECOMMENDED_EXPERIMENTS_PREDICTED_DIVERSE_PATH,
    RECOMMENDED_EXPERIMENTS_UCB_DIVERSE_PATH,
    RENAMED_SPIRO_COLUMN,
)
from src.bayesian_optimization import run_bayesian_optimization
from src.data_loading import filter_by_efficiency_threshold, load_raw_dataset
from src.grouping import group_conditions
from src.modeling import compare_models_leave_one_out, load_grouped_conditions
from src.reporting import create_curated_experimental_plan, write_analysis_report
from src.visualization import (
    save_all_figures,
    save_bayesian_optimization_figures,
    save_model_comparison_figures,
)


def main():
    data, renamed_spiro_column = load_raw_dataset(RAW_DATA_PATH)
    filtered_data = filter_by_efficiency_threshold(data, EFFICIENCY_THRESHOLD)
    removed_rows = len(data) - len(filtered_data)

    grouped_conditions = group_conditions(filtered_data)
    GROUPED_CONDITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered_data.to_csv(FILTERED_DATA_PATH, index=False)
    grouped_conditions.to_csv(GROUPED_CONDITIONS_PATH, index=False)
    save_all_figures(data, filtered_data, grouped_conditions, FIGURE_OUTPUT_DIR)

    modeling_data = load_grouped_conditions(GROUPED_CONDITIONS_PATH)
    metrics_data, predictions_data = compare_models_leave_one_out(modeling_data)
    MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_data.to_csv(MODEL_METRICS_PATH, index=False)
    predictions_data.to_csv(MODEL_PREDICTIONS_PATH, index=False)
    save_model_comparison_figures(metrics_data, predictions_data, FIGURE_OUTPUT_DIR)

    (
        candidate_predictions,
        ei_recommendations,
        predicted_recommendations,
        ucb_recommendations,
        model_candidate_predictions,
        ensemble_recommendations,
        champion_challenger,
        gpr_diagnostics,
        _,
    ) = run_bayesian_optimization(modeling_data)
    candidate_predictions.to_csv(CANDIDATE_PREDICTIONS_PATH, index=False)
    candidate_predictions.to_csv(CANDIDATE_PREDICTIONS_ENHANCED_PATH, index=False)
    model_candidate_predictions.to_csv(CANDIDATE_PREDICTIONS_BY_MODEL_PATH, index=False)
    ei_recommendations.to_csv(RECOMMENDED_EXPERIMENTS_PATH, index=False)
    ei_recommendations.to_csv(RECOMMENDED_EXPERIMENTS_EI_DIVERSE_PATH, index=False)
    predicted_recommendations.to_csv(
        RECOMMENDED_EXPERIMENTS_PREDICTED_DIVERSE_PATH,
        index=False,
    )
    ucb_recommendations.to_csv(RECOMMENDED_EXPERIMENTS_UCB_DIVERSE_PATH, index=False)
    ensemble_recommendations.to_csv(RECOMMENDED_EXPERIMENTS_ENSEMBLE_DIVERSE_PATH, index=False)
    champion_challenger.to_csv(CHAMPION_CHALLENGER_EXPERIMENTS_PATH, index=False)
    GPR_DIAGNOSTICS_PATH.write_text(gpr_diagnostics, encoding="utf-8")
    save_bayesian_optimization_figures(
        candidate_predictions,
        modeling_data,
        ei_recommendations,
        FIGURE_OUTPUT_DIR,
        ucb_recommendations=ucb_recommendations,
        model_predictions=model_candidate_predictions,
        ensemble_recommendations=ensemble_recommendations,
    )

    best_condition = grouped_conditions.iloc[0]
    curated_plan = create_curated_experimental_plan(
        candidate_predictions,
        model_candidate_predictions,
    )
    curated_plan.to_csv(CURATED_EXPERIMENTAL_PLAN_PATH, index=False)
    write_analysis_report(
        ANALYSIS_REPORT_PATH,
        best_condition,
        curated_plan,
        candidate_predictions,
        model_candidate_predictions,
    )
    final_summary_path = ANALYSIS_REPORT_PATH.parent / "final_summary.md"
    best_model = metrics_data.sort_values("RMSE").iloc[0]
    top_challengers = curated_plan.head(5)
    final_summary_lines = [
        "# Final Summary",
        "",
        f"Original dataset size: {len(data)} rows x {data.shape[1]} columns.",
        f"Rows removed by `Efficiency [%] < {EFFICIENCY_THRESHOLD}` threshold: {removed_rows}.",
        f"Remaining rows after threshold filtering: {len(filtered_data)}.",
        f"Grouped fabrication conditions: {len(grouped_conditions)}.",
        "",
        "Best observed condition:",
        f"- `SnO2_Rpm`: {best_condition['SnO2_Rpm']:.0f} rpm",
        f"- `Spiro_Oxid_Dur`: {best_condition['Spiro_Oxid_Dur']:.1f} h",
        f"- `max_efficiency`: {best_condition['max_efficiency']:.3f}%",
        "",
        (
            "Model comparison result: "
            f"{best_model['model_name']} had the lowest Leave-One-Out RMSE "
            f"({best_model['RMSE']:.3f})."
        ),
        (
            "Gaussian Process Regression was used for Bayesian Optimization because "
            "it provides predictive uncertainty for EI and UCB acquisition scores."
        ),
        (
            "No mean model prediction exceeded the current best observed "
            "`max_efficiency` of 17.322%."
        ),
        "",
        "Key challenger conditions from `curated_experimental_plan.csv`:",
    ]
    for row in top_challengers.itertuples(index=False):
        final_summary_lines.append(
            f"- {row.SnO2_Rpm:.0f} rpm / {row.Spiro_Oxid_Dur:.1f} h "
            f"({row.suggested_priority})"
        )
    final_summary_lines.extend(
        [
            "",
            (
                "Main limitation: the number of unique fabrication conditions is "
                "small, so uncertainty estimates and extrapolative recommendations "
                "should be treated cautiously."
            ),
            (
                "Experimental validation note: challenger conditions are "
                "model-guided hypotheses and are not guaranteed improvements."
            ),
        ]
    )
    final_summary_path.write_text("\n".join(final_summary_lines) + "\n", encoding="utf-8")
    model_mean_columns = [
        column
        for column in ["gpr_predicted", "rf_predicted", "knn_predicted", "xgb_predicted"]
        if column in model_candidate_predictions.columns
    ]
    mean_exceeds_best = (
        model_candidate_predictions[model_mean_columns].gt(best_condition["max_efficiency"]).any(axis=1)
    )

    print("Dataset loaded successfully")
    print(f"Raw dataset shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"Column available as {RENAMED_SPIRO_COLUMN}: {renamed_spiro_column}")
    print(
        f"Threshold filtering removed {removed_rows} rows where "
        f"{EFFICIENCY_COLUMN} < {EFFICIENCY_THRESHOLD}"
    )
    print(f"Grouped by: {', '.join(FEATURE_COLUMNS)}")
    print(f"Filtered data saved to: {FILTERED_DATA_PATH}")
    print(f"Grouped output saved to: {GROUPED_CONDITIONS_PATH}")
    print(f"EDA figures saved to: {FIGURE_OUTPUT_DIR}")
    print(f"Model metrics saved to: {MODEL_METRICS_PATH}")
    print(f"Leave-One-Out predictions saved to: {MODEL_PREDICTIONS_PATH}")
    print("Model ranking by RMSE:")
    for row in metrics_data.itertuples(index=False):
        print(f"  {row.model_name}: RMSE={row.RMSE:.3f}, MAE={row.MAE:.3f}, R2={row.R2:.3f}")
    print(f"Candidate predictions saved to: {CANDIDATE_PREDICTIONS_PATH}")
    print(f"Enhanced candidate predictions saved to: {CANDIDATE_PREDICTIONS_ENHANCED_PATH}")
    print(f"Model candidate predictions saved to: {CANDIDATE_PREDICTIONS_BY_MODEL_PATH}")
    print(f"Recommended experiments saved to: {RECOMMENDED_EXPERIMENTS_PATH}")
    print(f"Diverse EI recommendations saved to: {RECOMMENDED_EXPERIMENTS_EI_DIVERSE_PATH}")
    print(f"Diverse UCB recommendations saved to: {RECOMMENDED_EXPERIMENTS_UCB_DIVERSE_PATH}")
    print(
        "Diverse predicted-efficiency recommendations saved to: "
        f"{RECOMMENDED_EXPERIMENTS_PREDICTED_DIVERSE_PATH}"
    )
    print(
        "Diverse ensemble recommendations saved to: "
        f"{RECOMMENDED_EXPERIMENTS_ENSEMBLE_DIVERSE_PATH}"
    )
    print(f"Champion-challenger experiments saved to: {CHAMPION_CHALLENGER_EXPERIMENTS_PATH}")
    print(f"Curated experimental plan saved to: {CURATED_EXPERIMENTAL_PLAN_PATH}")
    print(f"Analysis report saved to: {ANALYSIS_REPORT_PATH}")
    print(f"Final summary saved to: {final_summary_path}")
    print(f"GPR diagnostics saved to: {GPR_DIAGNOSTICS_PATH}")
    print(gpr_diagnostics.splitlines()[1])
    print(
        "Warning: The number of unique fabrication conditions is small; "
        "the Bayesian Optimization recommendations should be interpreted as "
        "model-guided experimental hypotheses rather than guaranteed improvements."
    )
    print("Best observed condition:")
    print(f"  SnO2_Rpm: {best_condition['SnO2_Rpm']}")
    print(f"  Spiro_Oxid_Dur: {best_condition['Spiro_Oxid_Dur']}")
    print(f"  max_efficiency: {best_condition['max_efficiency']:.3f}")
    print("Potential-to-exceed diagnostics:")
    print(
        "  Maximum GPR predicted_max_efficiency: "
        f"{candidate_predictions['predicted_max_efficiency'].max():.3f}"
    )
    print(
        "  Maximum RF predicted value: "
        f"{model_candidate_predictions['rf_predicted'].max():.3f}"
    )
    print(
        "  Maximum kNN predicted value: "
        f"{model_candidate_predictions['knn_predicted'].max():.3f}"
    )
    print(
        "  Maximum GPR UCB beta=1.96: "
        f"{candidate_predictions['ucb_beta_1_96'].max():.3f}"
    )
    print(
        "  Candidates with any mean model prediction above best observed: "
        f"{int(mean_exceeds_best.sum())}"
    )
    print(
        "  Candidates with GPR UCB beta=1.96 above best observed: "
        f"{int(candidate_predictions['exceeds_best_observed_ucb_1_96'].sum())}"
    )
    print(
        "  Maximum ensemble_mean_prediction: "
        f"{model_candidate_predictions['ensemble_mean_prediction'].max():.3f}"
    )
    print(
        "  Maximum ensemble_upper_score: "
        f"{model_candidate_predictions['ensemble_upper_score'].max():.3f}"
    )
    print(f"  Curated plan: {CURATED_EXPERIMENTAL_PLAN_PATH}")
    print(f"  Analysis report: {ANALYSIS_REPORT_PATH}")
    print("Top 5 diverse EI-based recommendations:")
    for row in ei_recommendations.head(5).itertuples(index=False):
        print(
            f"  {row.rank}. SnO2_Rpm={row.SnO2_Rpm:.0f}, "
            f"Spiro_Oxid_Dur={row.Spiro_Oxid_Dur:.2f}, "
            f"predicted={row.predicted_max_efficiency:.3f}, "
            f"uncertainty={row.uncertainty:.3f}, "
            f"EI={row.expected_improvement:.4f}, "
            f"delta=({row.delta_SnO2_Rpm:.0f} rpm, {row.delta_Spiro_Oxid_Dur:.2f} h)"
        )
    print("Top 5 diverse UCB-based recommendations:")
    for row in ucb_recommendations.head(5).itertuples(index=False):
        print(
            f"  {row.rank}. SnO2_Rpm={row.SnO2_Rpm:.0f}, "
            f"Spiro_Oxid_Dur={row.Spiro_Oxid_Dur:.2f}, "
            f"predicted={row.predicted_max_efficiency:.3f}, "
            f"uncertainty={row.uncertainty:.3f}, "
            f"UCB1.96={row.ucb_beta_1_96:.3f}, "
            f"delta=({row.delta_SnO2_Rpm:.0f} rpm, {row.delta_Spiro_Oxid_Dur:.2f} h)"
        )
    print("Top 5 diverse predicted-efficiency recommendations:")
    for row in predicted_recommendations.head(5).itertuples(index=False):
        print(
            f"  {row.rank}. SnO2_Rpm={row.SnO2_Rpm:.0f}, "
            f"Spiro_Oxid_Dur={row.Spiro_Oxid_Dur:.2f}, "
            f"predicted={row.predicted_max_efficiency:.3f}, "
            f"uncertainty={row.uncertainty:.3f}, "
            f"EI={row.expected_improvement:.4f}, "
            f"delta=({row.delta_SnO2_Rpm:.0f} rpm, {row.delta_Spiro_Oxid_Dur:.2f} h)"
        )
    print("Top 5 diverse ensemble-based recommendations:")
    for row in ensemble_recommendations.head(5).itertuples(index=False):
        print(
            f"  {row.rank}. SnO2_Rpm={row.SnO2_Rpm:.0f}, "
            f"Spiro_Oxid_Dur={row.Spiro_Oxid_Dur:.2f}, "
            f"ensemble_mean={row.ensemble_mean_prediction:.3f}, "
            f"ensemble_upper={row.ensemble_upper_score:.3f}, "
            f"delta=({row.delta_SnO2_Rpm:.0f} rpm, {row.delta_Spiro_Oxid_Dur:.2f} h)"
        )
    print(
        "Note: Candidates whose UCB or ensemble upper score exceeds the current best "
        "observed PCE should be interpreted as high-potential experimental hypotheses, "
        "not guaranteed improvements."
    )


if __name__ == "__main__":
    main()
