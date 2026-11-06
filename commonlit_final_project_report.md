# CommonLit Readability Prize - Final Report

## Objective
Predict readability scores of English text excerpts.

## Dataset
Training excerpts: 2,834
Public example test excerpts: 7

## Data Preparation
- Missing text handling
- Word and character TF-IDF
- Text statistics
- Approximate readability formulas
- 80/20 train-validation split

## Initial Model Comparison

| Model | RMSE | MAE |
|---|---:|---:|
| Mean Baseline | 1.023588 | 0.831197 |
| Word TF-IDF + Ridge | 0.762301 | 0.619125 |
| Word + Char TF-IDF + Ridge | 0.698872 | 0.563424 |
| TF-IDF + Stats + Ridge | 0.679327 | 0.542914 |
| SVD + HGB | 0.676966 | 0.546093 |

## Hyperparameter Optimization
Ridge regularization (alpha):
0.5, 2.0, 6.0

Statistical feature weights:
0.15, 0.30, 0.50

Three additional HGB configurations evaluated.

## Final Selected Model
Ridge Regression

Alpha: 0.5
Statistical feature weight: 0.5

## Final Validation Results
RMSE: 0.655027
MAE: 0.521909

## Improvement
Initial best RMSE: 0.676966
Optimized RMSE: 0.655027
Relative RMSE reduction: approximately 3.24%

## Key Findings
TF-IDF and text statistics are effective
for modeling readability scores.

The optimized Ridge model outperformed
the tested SVD and HGB alternatives.

## Deliverables
- Trained model and text processors
- 7-row example Kaggle submission
- Streamlit readability prediction app
- Final report

## Limitations
- Single random validation split
- Hyperparameters selected using validation data
- Potentially optimistic validation score
- Small labeled dataset
- Only seven public example test excerpts
- No official Kaggle leaderboard score
- Approximate syllable and readability statistics

## Competition
https://www.kaggle.com/competitions/commonlitreadabilityprize
