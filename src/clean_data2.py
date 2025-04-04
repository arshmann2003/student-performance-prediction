import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(file_path, output_file_path):
    """
    This function loads the second student dataset, cleans it by handling missing values,
    removing duplicates, and ensuring correct data types. It then saves the cleaned dataset.

    Parameters:
    - file_path (str): Path to the CSV file.
    - output_file_path (str): Path to save the cleaned CSV file.

    Returns:
    - pd.DataFrame: Cleaned dataset.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Display initial info about the data
    print("Initial Data Info:")
    print(df.info())
    print("\nFirst 5 rows of the data:")
    print(df.head())

    # Handle missing values: Fill missing numerical values with the column mean
    df.fillna(df.mean(), inplace=True)

    # Check for and remove duplicates
    df.drop_duplicates(inplace=True)

    # Ensure correct data types (e.g., ensure 'Grades' is numeric)
    df['Grades'] = pd.to_numeric(df['Grades'], errors='coerce')

    # Check the cleaned data
    print("\nCleaned Data Info:")
    print(df.info())
    print("\nFirst 5 rows of the cleaned data:")
    print(df.head())

    # Save the cleaned data to a new file
    df.to_csv(output_file_path, index=False)
    print(f"\nCleaned data saved to {output_file_path}")

    return df

# Example usage
if __name__ == "__main__":
    file_path = "../data/raw_student_data2.csv"  # Update with actual file path
    output_file_path = "../data/cleaned_student_data2.csv"
    cleaned_data = load_and_clean_data(file_path, output_file_path)
