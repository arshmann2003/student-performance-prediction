from scripts.clean_merge import clean_and_save_data
from scripts.modeling_exam_score import train_exam_score_model
from scripts.gpa_random_forest import train_gpa_model
from scripts.clustering_student_profiles import run_clustering

def main():
    print("Step 1: Cleaning and Preparing Data")
    clean_and_save_data()

    print("Step 2: Training Exam Score Model")
    train_exam_score_model()

    print("Step 3: Training GPA Model")
    train_gpa_model()

    print("Step 4: Clustering Student Profiles")
    run_clustering()

    print("✅ All steps complete!")

if __name__ == "__main__":
    main()