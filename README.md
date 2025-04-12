# Student Performance Prediction with Machine Learning

This project analyzes student performance using machine learning techniques. It covers data cleaning, feature engineering, supervised learning (regression), unsupervised learning (clustering), and real-time GPA prediction based on input features. The work is organized into modular scripts for clarity and extensibility.

## 📁 Project Structure
```
student-sleep-ML/
├── data/
│   ├── raw/                  # Original datasets
│   ├── interim/              # Cleaned intermediate datasets
│   ├── input/                # Input files for prediction
│   └── output/               # Prediction outputs
├── figures/                  # Saved plots and graphs
├── models/                   # Trained models (pickle format)
├── scripts/                  # Core scripts
│   ├── clean_merge.py
│   ├── modeling_exam_score.py
│   ├── modeling_gpa.py
│   ├── clustering_student_profiles.py
│   └── predict_gpa.py
├── pipeline.py               # Main runner script
└── report.tex                # LaTeX report file
```

## 🔍 Features
- Data cleaning and encoding of multiple raw datasets
- Regression models to predict Exam Score and GPA
- Clustering of student profiles using KMeans and PCA
- Real-time GPA prediction based on three inputs
- Full academic report in LaTeX with analysis, figures, and insights

## 🧪 Setup & Installation
1. Clone the repository
```bash
git clone https://github.com/arshmann2003/student-sleep-ML.git
cd student-sleep-ML
```

2. Install dependencies (recommended in a virtual environment)
```bash
pip install -r requirements.txt
```

## 🚀 Usage
### Run the full pipeline:
```bash
python pipeline.py
```

### Predict GPA for new students:
```bash
python scripts/predict_gpa.py data/input/new_students.csv
```

## 📊 Input Format for GPA Prediction
Ensure your CSV has these columns:
```csv
StudyTimeWeekly,ParentalSupport,Absences
10.5,2,4
5.0,1,10
18.0,3,1
```

## 📄 Report
The full analysis is written in LaTeX and includes:
- Data overview
- Feature importance
- Statistical tests
- Modeling evaluation
- Clustering profiles

Output figures are saved in `/figures/`.

---

## ✍️ Author
Arshdeep Mann
Simon Fraser University

## 📌 Future Work
- Merge behavioral and socioeconomic data
- Apply SHAP for model explainability
- Improve model generalizability with more features