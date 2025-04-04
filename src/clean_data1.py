import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_clean_data(file_path, output_file):
    """
    Loads, cleans, and preprocesses student performance dataset.

    Parameters:
    - file_path (str): Path to the raw CSV file.
    - output_file (str): Path where cleaned data will be saved.

    Returns:
    - pd.DataFrame: Cleaned dataset.
    """

    # Load the dataset
    df = pd.read_csv(file_path)

    # Display initial info
    print("Initial Data Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())

    # Convert numerical columns explicitly to numeric type
    numeric_columns = [
        "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
        "Tutoring_Sessions", "Family_Income", "Physical_Activity", "Exam_Score"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert non-numeric to NaN

    # Handle missing values
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())  # Fill numeric NaN with mean

    # Categorical columns to fill missing values and encode
    categorical_columns = [
        "Parental_Involvement", "Access_to_Resources", "Extracurricular_Activities",
        "Motivation_Level", "Internet_Access", "Teacher_Quality", "School_Type",
        "Peer_Influence", "Learning_Disabilities", "Parental_Education_Level", "Gender"
    ]

    for col in categorical_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)  # Fill categorical NaN with mode

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    # Save cleaned data
    df.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to {output_file}")

    return df

if __name__ == "__main__":
    file_path = "../data/raw_student_data1.csv"
    output_file = "../data/cleaned_student_data1.csv"
    cleaned_data = load_and_clean_data(file_path, output_file)