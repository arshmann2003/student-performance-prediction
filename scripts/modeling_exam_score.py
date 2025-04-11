import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === Load the cleaned profile dataset ===
df = pd.read_csv('../data/interim/profile_cleaned.csv')

# === Encode categorical features to numeric ===
category_maps = {
    'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
    'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2}
}
for col, mapping in category_maps.items():
    df[col] = df[col].map(mapping)

# === Drop rows with any NaN values ===
df.dropna(inplace=True)

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
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_gbm, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Exam Score')
plt.ylabel('Predicted Exam Score')
plt.title('Gradient Boosting: Actual vs Predicted')
plt.grid(True)
plt.show()

# === Feature Importance from GBM ===
importances = pd.Series(gbm_model.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', title='Gradient Boosting Feature Importance')
plt.xlabel('Importance')
plt.grid(True)
plt.show()
