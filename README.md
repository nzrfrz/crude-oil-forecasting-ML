# Crude Oil Forecasting ML Pipeline

## Project Summary

This repository implements a WTI crude oil price forecasting pipeline using a combination of:
- data ingestion and macroeconomic feature merging,
- rolling LOWESS decomposition,
- CEEMDAN signal decomposition,
- feature engineering with lagged inputs,
- time-series model training with ensemble learning,
- unseen-date evaluation,
- interactive forecast and dataset update support.

The pipeline is designed for chronological forecasting with minimal data leakage and includes comprehensive statistical and graphical outputs for thesis-quality analysis.

## Key Files and Workflow

1. `01-data-ingestion-cleaning.py`
   - Downloads WTI crude oil futures and macroeconomic indicators via `yfinance`.
   - Merges data on trading dates, fills missing values, and saves `dataset/WTI-Macro-Master-UpToDate.csv`.

2. `02-rolling-lowess.py`
   - Applies rolling LOWESS decomposition to the price series using a 252-day window.
   - Produces trend and residual components.
   - Saves the output to `dataset/rolling-lowess-dataset.csv`.
   - Generates statistical and graphical evaluation in `evaluations/`.

3. `03-rolling-ceemdan.py`
   - Runs rolling CEEMDAN on the LOWESS residual to extract IMFs.
   - Reconstructs a denoised signal from IMF 4+.
   - Produces the machine learning master dataset at `dataset/master-features-dataset.csv`.
   - Saves additional evaluation plots and reports under `evaluations/`.

4. `03a-imf-justification.py`
   - Computes sample entropy and zero-crossing rate for extracted IMFs.
   - Creates a justification report for selecting noise vs. signal IMFs.
   - Produces visualizations in `evaluations/graphical/rolling-ceemdan`.

5. `04-fe-and-split.py`
   - Builds lagged features from price, trend, reconstructed signal, and macro indicators.
   - Applies winsorization and MinMax scaling.
   - Splits data chronologically into training, test, and unseen sets.
   - Saves split files in `dataset/splits/` and scalers in `dataset/scalers/`.

6. `05-model-training.py`
   - Loads the feature splits and trains multiple regressors:
     - `SVR`, `RandomForestRegressor`, `XGBRegressor`, and a `StackingRegressor` ensemble.
   - Performs time-series-aware hyperparameter tuning with `GridSearchCV`.
   - Saves trained models in `models/`.
   - Generates metrics, Diebold-Mariano comparisons, and prediction plots in `evaluations/`.

7. `evaluate_unseen.py`
   - Evaluates model forecasts on the held-out unseen dataset.
   - Saves unseen metrics and full comparison tables to `evaluations/statistical/unseen-eval/`.

8. `infere-engine.py`
   - Interactive engine for updating the dataset and forecasting new dates.
   - Supports incremental data updates and daily forecast output.
   - Writes forecast CSVs to `inference-output/`.
   - Note: this script references an auxiliary `fe-and-split-v2.py` file for incremental feature regeneration.

## Data and Outputs

- `dataset/` contains the raw merged data and processed feature datasets.
- `dataset/splits/` stores `X_train.csv`, `X_test.csv`, `X_unseen.csv`, `y_train.csv`, `y_test.csv`, and `y_unseen.csv`.
- `dataset/scalers/` stores fitted scalers used to inverse-transform predictions.
- `models/` contains the serialized trained model files.
- `evaluations/` contains statistical reports and graphical artifacts for each pipeline stage.
- `inference-output/` stores generated forecast CSVs and plots.

## Technologies Used

- Python
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `statsmodels`, `PyEMD`, `scipy`
- `scikit-learn`, `xgboost`, `joblib`
- `yfinance`

## How to Use

1. Run `01-data-ingestion-cleaning.py` to fetch raw data.
2. Run `02-rolling-lowess.py` to extract trend and residuals.
3. Run `03-rolling-ceemdan.py` to create the feature dataset.
4. Run `03a-imf-justification.py` for IMF noise/signal analysis.
5. Run `04-fe-and-split.py` to prepare scaled train/test/unseen splits.
6. Run `05-model-training.py` to train and save models.
7. Run `evaluate_unseen.py` to assess model performance on unseen data.
8. Optionally run `infere-engine.py` for interactive updates and forecasting.

## Notes

- The project is structured to preserve chronology and prevent look-ahead bias.
- The evaluation suite includes both model metrics and statistical significance testing.
- `infere-engine.py` provides a user-facing menu for dataset updates and forecasting.
