# Analysis Report

## Champion-Challenger Interpretation

The current best observed `max_efficiency` is 17.322%.

No mean model prediction exceeds this value: the GPR mean, Random Forest, kNN, and ensemble mean remain below the current champion.

GPR UCB identifies high-potential challenger candidates, but these are uncertainty-weighted hypotheses rather than confirmed improvements.

The top recommended validation batch is:
- 4200 rpm / 6.0 h
- 4050 rpm / 5.0 h
- 4300 rpm / 7.0 h
- 3600 rpm / 6.0 h
- 4650 rpm / 8.0 h

These conditions require experimental validation.

## Model Summary

Maximum GPR mean prediction: 16.262%
Maximum RF prediction: 16.907%
Maximum kNN prediction: 16.819%
Maximum ensemble mean prediction: 16.663%
Maximum ensemble upper score: 17.049%

## Curated Plan

- Rank 1 (High): 4200 rpm / 6.0 h. Strongest ensemble mean; interpolates between strong existing 4000/6 and 4500/6 region.
- Rank 2 (High): 4050 rpm / 5.0 h. Highest ensemble upper score; tests a slightly shorter oxidation duration near the high-performing 6 h region.
- Rank 3 (High): 4300 rpm / 7.0 h. High ensemble upper score; tests a slightly longer oxidation duration near the known 6 h optimum region.
- Rank 4 (Medium): 3600 rpm / 6.0 h. High GPR EI/UCB candidate; tests lower SnO2 rpm at the strong 6 h oxidation duration.
- Rank 5 (Medium): 4650 rpm / 8.0 h. Ensemble-supported medium-priority candidate at higher SnO2 rpm and moderate oxidation duration.
- Rank 6 (Medium): 4400 rpm / 9.0 h. Ensemble-supported medium-priority candidate extending the oxidation duration above 6-8 h.
- Rank 7 (Exploratory): 3500 rpm / 3.0 h. Exploratory short oxidation condition with UCB support but weaker ensemble support.
- Rank 8 (Exploratory): 4300 rpm / 17.5 h. Exploratory long-oxidation condition driven mainly by GPR uncertainty.
- Rank 9 (Exploratory): 4200 rpm / 18.5 h. Exploratory long-oxidation condition driven mainly by GPR uncertainty.
