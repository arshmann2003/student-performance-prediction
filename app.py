from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
BEHAVIOR_PATH = BASE_DIR / "data" / "interim" / "behavior_cleaned.csv"
PROFILE_PATH = BASE_DIR / "data" / "interim" / "profile_cleaned.csv"
FIGURES_DIR = BASE_DIR / "figures"

GPA_FEATURES = ["StudyTimeWeekly", "ParentalSupport", "Absences"]
EXAM_FEATURES = [
    "Attendance", "Hours_Studied", "Previous_Scores", "Sleep_Hours",
    "Tutoring_Sessions", "Motivation_Level", "Parental_Involvement",
    "Internet_Access", "Learning_Disabilities", "Family_Income",
]
LEVELS = {"Low": 0, "Medium": 1, "High": 2}
SUPPORT_LABELS = ["None", "Low", "Moderate", "High", "Very high"]

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")


@st.cache_data
def load_data():
    behavior = pd.read_csv(BEHAVIOR_PATH)
    profile = pd.read_csv(PROFILE_PATH)
    # Parental_Involvement is already numeric in the cleaned file; these two are still text.
    for col in ["Motivation_Level", "Family_Income"]:
        profile[col] = profile[col].map(LEVELS)
    profile = profile.dropna(subset=EXAM_FEATURES + ["Exam_Score"])
    return behavior, profile


def fit_and_score(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, {"r2": r2_score(y_test, pred), "mae": mean_absolute_error(y_test, pred)}


@st.cache_resource
def train_models():
    behavior, profile = load_data()
    gpa_model, gpa_metrics = fit_and_score(
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        behavior[GPA_FEATURES], behavior["GPA"],
    )
    exam_model, exam_metrics = fit_and_score(
        HistGradientBoostingRegressor(random_state=42),
        profile[EXAM_FEATURES], profile["Exam_Score"],
    )
    return gpa_model, gpa_metrics, exam_model, exam_metrics


def slider_for(df, col, label, step=1.0):
    lo, hi, mid = float(df[col].min()), float(df[col].max()), float(df[col].median())
    return st.slider(label, lo, hi, mid, step=step)


def sweep_chart(sweep, x_col, y_col, current_x, current_y):
    line = alt.Chart(sweep).mark_line().encode(
        x=alt.X(f"{x_col}:Q"),
        y=alt.Y(f"{y_col}:Q", scale=alt.Scale(zero=False)),
        tooltip=[x_col, alt.Tooltip(f"{y_col}:Q", format=".2f")],
    )
    current = pd.DataFrame({x_col: [current_x], y_col: [current_y]})
    rule = alt.Chart(current).mark_rule(color="red", strokeDash=[4, 4]).encode(x=f"{x_col}:Q")
    point = alt.Chart(current).mark_point(color="red", size=120, filled=True).encode(
        x=f"{x_col}:Q", y=f"{y_col}:Q",
    )
    st.altair_chart(line + rule + point, width="stretch")


behavior, profile = load_data()
gpa_model, gpa_metrics, exam_model, exam_metrics = train_models()

st.title("🎓 Student Performance Predictor")
st.caption("Adjust a student's habits and background to see their predicted GPA and exam score.")

gpa_tab, exam_tab, about_tab = st.tabs(["GPA", "Exam Score", "About"])

with gpa_tab:
    left, right = st.columns(2)
    with left:
        study = slider_for(behavior, "StudyTimeWeekly", "Study time (hours/week)", 0.5)
        support = st.select_slider(
            "Parental support", options=list(range(5)), value=2,
            format_func=lambda v: SUPPORT_LABELS[v],
        )
        absences = slider_for(behavior, "Absences", "Absences (days)")
    with right:
        x = pd.DataFrame([[study, support, absences]], columns=GPA_FEATURES)
        gpa = float(gpa_model.predict(x)[0])
        st.metric("Predicted GPA", f"{gpa:.2f} / 4.0")
        st.progress(min(max(gpa / 4.0, 0.0), 1.0))
        st.caption(
            f"Random Forest · test R² {gpa_metrics['r2']:.3f} · MAE {gpa_metrics['mae']:.2f}"
        )

        sweep = pd.DataFrame({
            "Absences": range(int(behavior["Absences"].min()), int(behavior["Absences"].max()) + 1),
        })
        sweep["StudyTimeWeekly"] = study
        sweep["ParentalSupport"] = support
        sweep["Predicted GPA"] = gpa_model.predict(sweep[GPA_FEATURES])
        st.markdown("**How absences change the prediction (other inputs held fixed)**")
        sweep_chart(sweep, "Absences", "Predicted GPA", absences, gpa)
        st.caption("Red marker = the current Absences value.")

with exam_tab:
    left, right = st.columns(2)
    with left:
        attendance = slider_for(profile, "Attendance", "Attendance (%)")
        hours = slider_for(profile, "Hours_Studied", "Hours studied (per week)")
        prev = slider_for(profile, "Previous_Scores", "Previous score")
        sleep = slider_for(profile, "Sleep_Hours", "Sleep (hours/night)")
        tutoring = slider_for(profile, "Tutoring_Sessions", "Tutoring sessions (per month)")
        motivation = st.select_slider("Motivation", options=list(LEVELS), value="Medium")
        involvement = st.select_slider("Parental involvement", options=list(LEVELS), value="Medium")
        income = st.select_slider("Family income", options=list(LEVELS), value="Medium")
        internet = st.toggle("Internet access", value=True)
        disability = st.toggle("Learning disability", value=False)
    with right:
        x = pd.DataFrame([[
            attendance, hours, prev, sleep, tutoring, LEVELS[motivation],
            LEVELS[involvement], int(internet), int(disability), LEVELS[income],
        ]], columns=EXAM_FEATURES)
        score = float(exam_model.predict(x)[0])
        st.metric("Predicted exam score", f"{score:.1f}")
        st.caption(
            f"Gradient Boosting · test R² {exam_metrics['r2']:.3f} · MAE {exam_metrics['mae']:.2f}"
        )

        lo, hi = int(profile["Attendance"].min()), int(profile["Attendance"].max())
        sweep = pd.concat([x] * (hi - lo + 1), ignore_index=True)
        sweep["Attendance"] = range(lo, hi + 1)
        sweep["Predicted score"] = exam_model.predict(sweep[EXAM_FEATURES])
        st.markdown("**How attendance changes the prediction (other inputs held fixed)**")
        sweep_chart(sweep, "Attendance", "Predicted score", attendance, score)
        st.caption("Red marker = the current Attendance value.")

with about_tab:
    st.markdown(
        f"""
        Two models trained on public Kaggle student datasets:

        - **GPA**: Random Forest on weekly study time, parental support, and absences
          ({len(behavior):,} students).
        - **Exam score**: Histogram Gradient Boosting on attendance, study hours, prior
          scores, sleep, tutoring, motivation, and family factors ({len(profile):,} students).

        Predictions reflect associations in this data, not causal effects.

        Source code: [GitHub](https://github.com/arshmann2003/student-performance-prediction)
        """
    )
    c1, c2 = st.columns(2)
    c1.image(str(FIGURES_DIR / "gpa_feature_importances.png"), caption="GPA feature importance")
    c2.image(str(FIGURES_DIR / "clustering_student_profiles.png"), caption="Student clusters (PCA)")
