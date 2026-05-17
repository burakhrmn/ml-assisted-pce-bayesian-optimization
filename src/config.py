from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
FILTERED_DATA_PATH = PROJECT_ROOT / "outputs" / "tables" / "filtered_data.csv"
GROUPED_CONDITIONS_PATH = PROJECT_ROOT / "outputs" / "tables" / "grouped_conditions.csv"
MODEL_METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "model_comparison_metrics.csv"
MODEL_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "tables" / "model_predictions_leave_one_out.csv"
CANDIDATE_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "tables" / "candidate_predictions.csv"
CANDIDATE_PREDICTIONS_ENHANCED_PATH = (
    PROJECT_ROOT / "outputs" / "tables" / "candidate_predictions_enhanced.csv"
)
CANDIDATE_PREDICTIONS_BY_MODEL_PATH = (
    PROJECT_ROOT / "outputs" / "tables" / "candidate_predictions_by_model.csv"
)
RECOMMENDED_EXPERIMENTS_PATH = PROJECT_ROOT / "outputs" / "tables" / "recommended_experiments.csv"
RECOMMENDED_EXPERIMENTS_EI_DIVERSE_PATH = (
    PROJECT_ROOT / "outputs" / "tables" / "recommended_experiments_ei_diverse.csv"
)
RECOMMENDED_EXPERIMENTS_PREDICTED_DIVERSE_PATH = (
    PROJECT_ROOT / "outputs" / "tables" / "recommended_experiments_predicted_diverse.csv"
)
RECOMMENDED_EXPERIMENTS_UCB_DIVERSE_PATH = (
    PROJECT_ROOT / "outputs" / "tables" / "recommended_experiments_ucb_diverse.csv"
)
RECOMMENDED_EXPERIMENTS_ENSEMBLE_DIVERSE_PATH = (
    PROJECT_ROOT / "outputs" / "tables" / "recommended_experiments_ensemble_diverse.csv"
)
CHAMPION_CHALLENGER_EXPERIMENTS_PATH = (
    PROJECT_ROOT / "outputs" / "tables" / "champion_challenger_experiments.csv"
)
CURATED_EXPERIMENTAL_PLAN_PATH = (
    PROJECT_ROOT / "outputs" / "tables" / "curated_experimental_plan.csv"
)
ANALYSIS_REPORT_PATH = PROJECT_ROOT / "outputs" / "analysis_report.md"
GPR_DIAGNOSTICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "gpr_diagnostics.txt"
FIGURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

FEATURE_COLUMNS = ["SnO2_Rpm", "Spiro_Oxid_Dur"]
TARGET_COLUMN = "max_efficiency"
EFFICIENCY_COLUMN = "Efficiency [%]"
ORIGINAL_SPIRO_COLUMN = "Spiro_Des"
RENAMED_SPIRO_COLUMN = "Spiro_Oxid_Dur"
EFFICIENCY_THRESHOLD = 11
RANDOM_SEED = 42

SNO2_RPM_BOUNDS = (3500, 5000)
SPIRO_OXID_DUR_BOUNDS = (3, 24)
SNO2_RPM_GRID_STEP = 50
SPIRO_OXID_DUR_GRID_STEP = 0.5
MIN_RECOMMENDATION_RPM_SEPARATION = 100
MIN_RECOMMENDATION_SPIRO_SEPARATION = 1.0
MIN_EXISTING_RPM_SEPARATION = 100
MIN_EXISTING_SPIRO_SEPARATION = 1.0
EXPECTED_IMPROVEMENT_XI = 0.01
N_RECOMMENDATIONS = 20
