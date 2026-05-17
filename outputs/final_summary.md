# Final Summary

Original dataset size: 456 rows x 11 columns.
Rows removed by `Efficiency [%] < 11` threshold: 50.
Remaining rows after threshold filtering: 406.
Grouped fabrication conditions: 19.

Best observed condition:
- `SnO2_Rpm`: 4500 rpm
- `Spiro_Oxid_Dur`: 6.0 h
- `max_efficiency`: 17.322%

Model comparison result: Random Forest Regressor had the lowest Leave-One-Out RMSE (0.936).
Gaussian Process Regression was used for Bayesian Optimization because it provides predictive uncertainty for EI and UCB acquisition scores.
No mean model prediction exceeded the current best observed `max_efficiency` of 17.322%.

Key challenger conditions from `curated_experimental_plan.csv`:
- 4200 rpm / 6.0 h (High)
- 4050 rpm / 5.0 h (High)
- 4300 rpm / 7.0 h (High)
- 3600 rpm / 6.0 h (Medium)
- 4650 rpm / 8.0 h (Medium)

Main limitation: the number of unique fabrication conditions is small, so uncertainty estimates and extrapolative recommendations should be treated cautiously.
Experimental validation note: challenger conditions are model-guided hypotheses and are not guaranteed improvements.
