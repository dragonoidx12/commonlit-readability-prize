# CommonLit Readability Prize

English text readability prediction using
TF-IDF, text statistics and Ridge Regression.

## Final Model
Ridge alpha=0.5
Statistical feature weight=0.5

## Validation
RMSE: 0.655027
MAE: 0.521909

## Streamlit Features
- Enter an English text and predict readability
- View simple text statistics
- Upload CSV for batch predictions
- Download prediction results

## Installation
pip install -r requirements.txt

## Start
streamlit run app.py

## Notes
The model estimates the CommonLit target score.
It does not directly estimate reading grade level.
Use a compatible Python/scikit-learn environment.

## Competition
https://www.kaggle.com/competitions/commonlitreadabilityprize
