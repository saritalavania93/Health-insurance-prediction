import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import joblib
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "/content/insurance.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"CSV file not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

expected_cols = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
missing_cols = [c for c in expected_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"Dataset is missing required columns: {missing_cols}")

df = df[expected_cols].copy()

num_cols = ["age", "bmi", "children", "charges"]
cat_cols = ["sex", "smoker", "region"]

for c in num_cols:
    if df[c].isna().sum() > 0:
        df[c] = df[c].fillna(df[c].median())

for c in cat_cols:
    if df[c].isna().sum() > 0:
        df[c] = df[c].fillna(df[c].mode()[0])

X = df.drop("charges", axis=1)
y = df["charges"]

numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

def adjusted_r2(r2, n, p):
    if n <= p + 1:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def regression_metrics(y_true, y_pred, n, p):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adjr2 = adjusted_r2(r2, n, p)
    return mae, mse, rmse, r2, adjr2

def evaluate_model(name, model, X_train, y_train, X_test, y_test, preprocessor):
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    yhat_train = pipe.predict(X_train)
    yhat_test = pipe.predict(X_test)

    Xt_train = pipe.named_steps["preprocess"].transform(X_train)
    p = Xt_train.shape[1]
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    train_mae, train_mse, train_rmse, train_r2, train_adj = regression_metrics(y_train, yhat_train, n_train, p)
    test_mae, test_mse, test_rmse, test_r2, test_adj = regression_metrics(y_test, yhat_test, n_test, p)

    overfit = "Y" if (train_r2 - test_r2) > 0.10 else "N"

    return {
        "Model": name,
        "Test RMSE": test_rmse,
        "Test R2": test_r2,
        "Pipeline": pipe
    }

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
    "Extra Trees": ExtraTreesRegressor(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostRegressor(random_state=RANDOM_STATE),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor()
}

results = []
for name, model in models.items():
    res = evaluate_model(name, model, X_train, y_train, X_test, y_test, preprocessor)
    results.append(res)

results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "Pipeline"} for r in results])
results_df = results_df.sort_values("Test RMSE").reset_index(drop=True)

best_model_name = results_df.iloc[0]["Model"]
print("Best baseline model:", best_model_name)

best_pipeline = None
for r in results:
    if r["Model"] == best_model_name:
        best_pipeline = r["Pipeline"]
        break

MODEL_PATH = "best_insurance_model.joblib"
joblib.dump(best_pipeline, MODEL_PATH)
print(f"Saved best model to: {MODEL_PATH}")
