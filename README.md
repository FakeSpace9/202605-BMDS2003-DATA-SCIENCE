# Gold Price Prediction (BMDS2003 Data Science)

This repository contains an end-to-end machine learning project designed to predict daily gold prices using historical market data. Following the CRISP-DM methodology, the project includes extensive feature engineering, walk-forward model validation to prevent data leakage, and a Streamlit dashboard for interactive forecasting.

## 📁 Project Structure

````text
202605-BMDS2003-DATA-SCIENCE/
│
├── data/
│   ├── raw/
│   │   └── Gold_Price.csv                      # Raw historical dataset
│   └── processed/
│       ├── Gold_Price_cleaned.csv              # Cleaned/engineered dataset
│       ├── Gold_Price_train.csv                # Training split
│       └── Gold_Price_test.csv                 # Testing split
│
├── report_assets/
│   ├── plots/                                  # 15+ generated EDA visualizations
│   └── 202605 BMDS2003 Assignment Specification (1).pdf
│   └── 202605 BMDS2003 Marking Rubrics (1).pdf
│
├── src/
│   ├── eda_visuals.py                          # Script to generate exploratory graphs
│   ├── model_evaluation_visual.py              # Baseline evaluation and scatter plots
│   ├── preprocessing.py                        # Handles missing data, outliers, scaling, and feature engineering
│   ├── utils.py                                # Shared helper functions for metrics and file loading
│   └── models/
│       ├── model_training.py                   # Standard train/test split model training
│       ├── linear_reg_walkforward.py           # Expanding-window walk-forward Linear Regression
│       ├── train_gradient_boosting.py          # Walk-forward Gradient Boosting (Log Return)
│       ├── train_knn.py                        # Walk-forward KNN Regression (Daily Difference)
│       └── train_random_forest.py              # Walk-forward Random Forest (Log Return)
│
├── prototype/                                  # Deployment and Application directory
│   ├── app.py                                  # Streamlit web dashboard
│   ├── scaler.pkl                              # Fitted StandardScaler for inference
│   ├── summary_metrics/                        # JSON performance metrics for each model
│   ├── fold_metrics/                           # CSV files containing fold-by-fold results
│   ├── feature_coefficient/                    # Extracted feature importances/coefficients
│   └── model/                                  # Saved .pkl model files
│
└── README.md

````text
## ⚙️ Installation & Prerequisites

To set up the environment, run this command in your terminal to install all the required dependencies for the models and the dashboard[cite: 1]:

```bash
pip install pandas matplotlib seaborn scikit-learn streamlit numpy plotly joblib

## 🚀 Execution Order

To run the project properly from scratch, execute the following commands in the terminal in this exact sequence:

### Step 1: Data Preprocessing
Clean the raw data, handle missing values, and engineer predictive features (Lags, Rolling Volatility, RSI, MACD, etc.).
```bash
python src/preprocessing.py

### Step 2: Exploratory Data Analysis (EDA)
Generate the analytical plots (automatically saved to report_assets/plots/).
```bash
python src/eda_visuals.py

### Step 3: Model Training
Train the predictive models. Run the baseline training first, followed by the walk-forward validation scripts.
```bash
python src/models/linear_reg_walkforward.py
python src/models/train_knn.py
python src/models/train_random_forest.py
python src/models/train_gradient_boosting.py

### Step 4: Launch the Web Application
Launch the interactive Streamlit dashboard to view metrics and make forecasts.
```bash
streamlit run prototype/app.py
````
