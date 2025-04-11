import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import joblib
import os

# === File paths ===
PROFILE_PATH = 'data/interim/profile_cleaned.csv'

def train_exam_score_model():
    # === Load the cleaned profile dataset ===
    df = pd.read_csv(PROFILE_PATH)

    # === Encode categorical features to numeric ===
    category_maps = {
        'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
        'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
        'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2}
    }
    for col, mapping in category_maps.items():
        df[col] = df[col].map(mapping).fillna(-1)

    # === Feature selection ===
    features = [
        'Attendance', 'Hours_Studied', 'Previous_Scores',
        'Sleep_Hours', 'Tutoring_Sessions', 'Motivation_Level',
        'Parental_Involvement', 'Internet_Access', 'Learning_Disabilities',
        'Family_Income'
    ]
    target = 'Exam_Score'

    X = df[features]
    y = df[target]

    # === Train-test split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Linear Regression ===
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_pred_lin = lin_model.predict(X_test)

    # === HistGradientBoosting Regression (GBM) ===
    gbm_model = HistGradientBoostingRegressor(random_state=42)
    gbm_model.fit(X_train, y_train)
    y_pred_gbm = gbm_model.predict(X_test)

    # === Save GBM model ===
    os.makedirs("../models", exist_ok=True)
    joblib.dump(gbm_model, "../models/gbm_exam_score_model.pkl")
    print("✅ GBM model saved to ../models/gbm_exam_score_model.pkl")

    # === Evaluation Function ===
    def evaluate_model(name, y_true, y_pred):
        print(f"\n{name} Performance:")
        print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
        print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
        print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"RMSE: {rmse:.2f}")

    # === Evaluate Models ===
    evaluate_model("Linear Regression", y_test, y_pred_lin)
    evaluate_model("Gradient Boosting", y_test, y_pred_gbm)

    # === Cross-Validation for GBM ===
    cv_scores = cross_val_score(gbm_model, X, y, cv=5, scoring='r2')
    print(f"\nCross-validated R² (GBM): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # === Plot Actual vs Predicted (GBM) ===
    os.makedirs("../figures", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred_gbm, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Exam Score')
    plt.ylabel('Predicted Exam Score')
    plt.title('Gradient Boosting: Actual vs Predicted')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../figures/gbm_exam_pred_vs_actual.png")
    plt.show()

    # === Permutation Feature Importance for GBM ===
    result = permutation_importance(gbm_model, X_test, y_test, n_repeats=10, random_state=42)
    importances = pd.Series(result.importances_mean, index=features)
    importances.sort_values().plot(kind='barh', title='Permutation Feature Importance')
    plt.xlabel('Importance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../figures/gbm_exam_feature_importance.png")
    plt.show()
