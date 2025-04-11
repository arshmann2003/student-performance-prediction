import pandas as pd
import os

def clean_and_save_data():
    raw_profile_path = 'data/raw/student_profile_data.csv'
    raw_socio_path = 'data/raw/socioeconomic_scores.csv'
    raw_behavior_path = 'data/raw/academic_behavioural_data.csv'

    profile_df = pd.read_csv(raw_profile_path)
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

    socio_df = pd.read_csv(raw_socio_path)
    socio_df.ffill(inplace=True)
    socio_df.bfill(inplace=True)
    socio_df.to_csv('../data/interim/socio_cleaned.csv', index=False)
    print("✅ Cleaned socioeconomic data saved to ../data/interim/socio_cleaned.csv")

    behavior_df = pd.read_csv(raw_behavior_path)
    behavior_df.rename(columns={'GPA': 'GPA', 'GradeClass': 'Grade_Class'}, inplace=True)
    behavior_df.ffill(inplace=True)
    behavior_df.bfill(inplace=True)
    behavior_df.to_csv('../data/interim/behavior_cleaned.csv', index=False)
    print("✅ Cleaned behavioral data saved to ../data/interim/behavior_cleaned.csv")
