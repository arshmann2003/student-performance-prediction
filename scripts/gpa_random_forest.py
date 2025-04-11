import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

BEHAVIOR_PATH = "data/interim/behavior_cleaned.csv"


def train_gpa_model():
    # Load the cleaned behavioral dataset
    data = pd.read_csv(BEHAVIOR_PATH)

    # Select three features and target
    features = ["StudyTimeWeekly", "ParentalSupport", "Absences"]
    X = data[features]
    y = data["GPA"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train regressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)

    # Save trained model
    os.makedirs("../models", exist_ok=True)
    joblib.dump(regressor, "../models/gpa_random_forest_model.pkl")
    print("✅ Model saved to ../models/gpa_random_forest_model.pkl")

    # Evaluate on test set
    y_pred = regressor.predict(X_test)
    print("\nRegression Performance on Test Set:")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    # Plot feature importances
    os.makedirs("../figures", exist_ok=True)
    importances = pd.Series(regressor.feature_importances_, index=features)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Feature Importances - Random Forest Regressor")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("../figures/gpa_feature_importances.png")
    plt.show()

    # Plot predicted vs actual
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')
    plt.xlabel("Actual GPA")
    plt.ylabel("Predicted GPA")
    plt.title("Actual vs Predicted GPA")
    plt.tight_layout()
    plt.savefig("../figures/gpa_predicted_vs_actual.png")
    plt.show()

    # --- Predict on new input file (3 features only) ---
    new_input_path = "data/input/new_students.csv"
    new_data = pd.read_csv(new_input_path)
    X_new = new_data[features]

    predictions = regressor.predict(X_new)
    new_data["Predicted_GPA"] = predictions
    print("\n📋 New Predictions:")
    print(new_data)

    # Save to file
    output_path = "data/output/predicted_gpa.csv"
    os.makedirs("data/output", exist_ok=True)
    new_data.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
