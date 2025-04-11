import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# === Base directory relative to this script ===
BASE_DIR = Path(__file__).resolve().parent.parent

# === Load model ===
model_path = BASE_DIR / "models" / "gpa_random_forest_model.pkl"
regressor = joblib.load(model_path)
print(f"✅ Loaded model from {model_path}")

# === Features used during training ===
features = ["StudyTimeWeekly", "ParentalSupport", "Absences"]

# === Take input path from command-line ===
if len(sys.argv) != 2:
    print("❌ Usage: python predict_gpa.py <input_csv_path>")
    sys.exit(1)

input_path = Path(sys.argv[1])
if not input_path.exists():
    print(f"❌ File not found: {input_path}")
    sys.exit(1)

# === Load input data and predict ===
new_data = pd.read_csv(input_path)
X_new = new_data[features]

predictions = regressor.predict(X_new)
new_data["Predicted_GPA"] = predictions
print("\n📋 New Predictions:")
print(new_data)

# === Save predictions ===
output_dir = BASE_DIR / "predict" / "output"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "predicted_gpa.csv"
new_data.to_csv(output_path, index=False)
print(f"✅ Predictions saved to {output_path}")