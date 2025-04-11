import pandas as pd
import os

# === Create output folder if it doesn't exist ===
os.makedirs('../data/interim', exist_ok=True)

# === Load and clean profile_df ===
profile_df = pd.read_csv('../data/raw/student_profile_data.csv')
profile_df['Parental_Involvement'] = profile_df['Parental_Involvement'].map({'Low': 0, 'Medium': 1, 'High': 2})
profile_df['Access_to_Resources'] = profile_df['Access_to_Resources'].map({'Low': 0, 'Medium': 1, 'High': 2})
profile_df['Extracurricular_Activities'] = profile_df['Extracurricular_Activities'].map({'No': 0, 'Yes': 1})
profile_df['Internet_Access'] = profile_df['Internet_Access'].map({'No': 0, 'Yes': 1})
profile_df['Learning_Disabilities'] = profile_df['Learning_Disabilities'].map({'No': 0, 'Yes': 1})
profile_df['Gender'] = profile_df['Gender'].map({'Male': 1, 'Female': 0})
profile_df.ffill(inplace=True)
profile_df.bfill(inplace=True)
profile_df.to_csv('../data/interim/profile_cleaned.csv', index=False)
print("✅ Cleaned profile data saved to ../data/interim/profile_cleaned.csv")

# === Load and clean socio_df ===
socio_df = pd.read_csv('../data/raw/socioeconomic_scores.csv')
socio_df.ffill(inplace=True)
socio_df.bfill(inplace=True)
socio_df.to_csv('../data/interim/socio_cleaned.csv', index=False)
print("✅ Cleaned socioeconomic data saved to ../data/interim/socio_cleaned.csv")

# === Load and clean behavior_df ===
behavior_df = pd.read_csv('../data/raw/academic_behavioural_data.csv')
behavior_df.rename(columns={'GPA': 'GPA', 'GradeClass': 'Grade_Class'}, inplace=True)
behavior_df.ffill(inplace=True)
behavior_df.bfill(inplace=True)
behavior_df.to_csv('../data/interim/behavior_cleaned.csv', index=False)
print("✅ Cleaned behavioral data saved to ../data/interim/behavior_cleaned.csv")