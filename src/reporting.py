import pandas as pd


CURATED_CONDITIONS = [
    {
        "priority_rank": 1,
        "suggested_priority": "High",
        "SnO2_Rpm": 4200.0,
        "Spiro_Oxid_Dur": 6.0,
        "experimental_rationale": (
            "Strongest ensemble mean; interpolates between strong existing "
            "4000/6 and 4500/6 region."
        ),
    },
    {
        "priority_rank": 2,
        "suggested_priority": "High",
        "SnO2_Rpm": 4050.0,
        "Spiro_Oxid_Dur": 5.0,
        "experimental_rationale": (
            "Highest ensemble upper score; tests a slightly shorter oxidation "
            "duration near the high-performing 6 h region."
        ),
    },
    {
        "priority_rank": 3,
        "suggested_priority": "High",
        "SnO2_Rpm": 4300.0,
        "Spiro_Oxid_Dur": 7.0,
        "experimental_rationale": (
            "High ensemble upper score; tests a slightly longer oxidation "
            "duration near the known 6 h optimum region."
        ),
    },
    {
        "priority_rank": 4,
        "suggested_priority": "Medium",
        "SnO2_Rpm": 3600.0,
        "Spiro_Oxid_Dur": 6.0,
        "experimental_rationale": (
            "High GPR EI/UCB candidate; tests lower SnO2 rpm at the strong "
            "6 h oxidation duration."
        ),
    },
    {
        "priority_rank": 5,
        "suggested_priority": "Medium",
        "SnO2_Rpm": 4650.0,
        "Spiro_Oxid_Dur": 8.0,
        "experimental_rationale": (
            "Ensemble-supported medium-priority candidate at higher SnO2 rpm "
            "and moderate oxidation duration."
        ),
    },
    {
        "priority_rank": 6,
        "suggested_priority": "Medium",
        "SnO2_Rpm": 4400.0,
        "Spiro_Oxid_Dur": 9.0,
        "experimental_rationale": (
            "Ensemble-supported medium-priority candidate extending the "
            "oxidation duration above 6-8 h."
        ),
    },
    {
        "priority_rank": 7,
        "suggested_priority": "Exploratory",
        "SnO2_Rpm": 3500.0,
        "Spiro_Oxid_Dur": 3.0,
        "experimental_rationale": (
            "Exploratory short oxidation condition with UCB support but weaker "
            "ensemble support."
        ),
    },
    {
        "priority_rank": 8,
        "suggested_priority": "Exploratory",
        "SnO2_Rpm": 4300.0,
        "Spiro_Oxid_Dur": 17.5,
        "experimental_rationale": (
            "Exploratory long-oxidation condition driven mainly by GPR uncertainty."
        ),
    },
    {
        "priority_rank": 9,
        "suggested_priority": "Exploratory",
        "SnO2_Rpm": 4200.0,
        "Spiro_Oxid_Dur": 18.5,
        "experimental_rationale": (
            "Exploratory long-oxidation condition driven mainly by GPR uncertainty."
        ),
    },
]


def create_curated_experimental_plan(enhanced_candidates, model_candidates):
    """Create the final curated validation plan from candidate prediction tables."""
    curated = pd.DataFrame(CURATED_CONDITIONS)
    model_columns = [
        column for column in model_candidates.columns if column != "gpr_predicted"
    ]
    merged = curated.merge(
        enhanced_candidates,
        on=["SnO2_Rpm", "Spiro_Oxid_Dur"],
        how="inner",
    ).merge(
        model_candidates[model_columns],
        on=["SnO2_Rpm", "Spiro_Oxid_Dur"],
        how="left",
    )

    merged = merged.rename(
        columns={
            "predicted_max_efficiency": "gpr_predicted",
            "uncertainty": "gpr_uncertainty",
        }
    )
    output_columns = [
        "priority_rank",
        "suggested_priority",
        "SnO2_Rpm",
        "Spiro_Oxid_Dur",
        "gpr_predicted",
        "gpr_uncertainty",
        "expected_improvement",
        "ucb_beta_1_96",
        "rf_predicted",
        "knn_predicted",
        "ensemble_mean_prediction",
        "ensemble_upper_score",
        "exceeds_best_observed_prediction",
        "exceeds_best_observed_ucb_1_96",
        "exceeds_best_observed_ensemble_mean",
        "exceeds_best_observed_ensemble_upper",
        "experimental_rationale",
    ]
    optional_columns = [column for column in output_columns if column in merged.columns]
    return merged[optional_columns].sort_values("priority_rank").reset_index(drop=True)


def write_analysis_report(path, best_condition, curated_plan, candidate_predictions, model_predictions):
    """Write a concise Markdown interpretation report for the final validation plan."""
    best_observed = float(best_condition["max_efficiency"])
    top_batch = curated_plan.head(5)
    lines = [
        "# Analysis Report",
        "",
        "## Champion-Challenger Interpretation",
        "",
        f"The current best observed `max_efficiency` is {best_observed:.3f}%.",
        "",
        (
            "No mean model prediction exceeds this value: the GPR mean, Random "
            "Forest, kNN, and ensemble mean remain below the current champion."
        ),
        "",
        (
            "GPR UCB identifies high-potential challenger candidates, but these "
            "are uncertainty-weighted hypotheses rather than confirmed improvements."
        ),
        "",
        "The top recommended validation batch is:",
    ]
    for row in top_batch.itertuples(index=False):
        lines.append(f"- {row.SnO2_Rpm:.0f} rpm / {row.Spiro_Oxid_Dur:.1f} h")

    lines.extend(
        [
            "",
            "These conditions require experimental validation.",
            "",
            "## Model Summary",
            "",
            (
                f"Maximum GPR mean prediction: "
                f"{candidate_predictions['predicted_max_efficiency'].max():.3f}%"
            ),
            (
                f"Maximum RF prediction: "
                f"{model_predictions['rf_predicted'].max():.3f}%"
            ),
            (
                f"Maximum kNN prediction: "
                f"{model_predictions['knn_predicted'].max():.3f}%"
            ),
            (
                f"Maximum ensemble mean prediction: "
                f"{model_predictions['ensemble_mean_prediction'].max():.3f}%"
            ),
            (
                f"Maximum ensemble upper score: "
                f"{model_predictions['ensemble_upper_score'].max():.3f}%"
            ),
            "",
            "## Curated Plan",
            "",
        ]
    )
    for row in curated_plan.itertuples(index=False):
        lines.append(
            f"- Rank {row.priority_rank} ({row.suggested_priority}): "
            f"{row.SnO2_Rpm:.0f} rpm / {row.Spiro_Oxid_Dur:.1f} h. "
            f"{row.experimental_rationale}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
