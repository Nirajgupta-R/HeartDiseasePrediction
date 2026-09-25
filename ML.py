import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.stApp {
    background: #f6f8fc;
}

/* Model Information text */
div[data-testid="stMarkdownContainer"] {
    color: #263b62 !important;
}

div[data-testid="stMarkdownContainer"] p {
    color: #263b62 !important;
}

div[data-testid="stMarkdownContainer"] strong {
    color: #263b62 !important;
}

h1, h2, h3, h4 {
    color: #263b62 !important;
}


/* ================================
   SIDEBAR
================================ */

[data-testid="stSidebar"] {
    background-color: #1f2d50 !important;
}

[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
    color: #ffffff !important;
}

[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
}

[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] strong {
    color: #ffffff !important;
}

[data-testid="stSidebar"] label {
    color: #ffffff !important;
}

[data-testid="stSidebar"] label p {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #ffffff !important;
}


/* ---------------- TOP HEADER ---------------- */

.top-header {
    background: linear-gradient(135deg, #18294b, #26395f);
    padding: 18px 28px;
    border-radius: 0 0 12px 12px;
    color: white;
    margin-bottom: 15px;
}

.logo-area {
    display: flex;
    align-items: center;
    gap: 14px;
}

.heart-logo {
    width: 46px;
    height: 46px;
    background: #ff6b81;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
}

.header-title {
    font-size: 23px;
    font-weight: 700;
}

.header-subtitle {
    font-size: 15px;
    opacity: 0.85;
}

/* ---------------- SIDEBAR ---------------- */

section[data-testid="stSidebar"] {
    background: #18294b;
}

section[data-testid="stSidebar"] > div {
    background: #18294b;
}

.sidebar-title {
    color: white;
    font-size: 18px;
    font-weight: 700;
    padding: 10px 10px 20px 10px;
}

.sidebar-card {
    background: #21375f;
    padding: 15px;
    border-radius: 9px;
    margin-top: 20px;
    color: white;
}

.sidebar-card-title {
    font-weight: 700;
    font-size: 15px;
}

.sidebar-card-text {
    font-size: 12px;
    color: #d3dbea;
    line-height: 1.5;
}

/* ---------------- CARDS ---------------- */

.card {
    background: white;
    border: 1px solid #e2e7f0;
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 15px;
    box-shadow: 0 2px 8px rgba(30, 50, 90, 0.04);
}

.card-title {
    color: #20345d;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 5px;
}

.card-subtitle {
    color: #71809c;
    font-size: 11px;
}

/* ---------------- RESULT ---------------- */

.result-success {
    background: linear-gradient(90deg, #edf9f1, #f7fcf8);
    border: 1px solid #cdebd6;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

.result-danger {
    background: linear-gradient(90deg, #fff0f2, #fff8f8);
    border: 1px solid #f3ccd3;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

.result-title {
    font-size: 15px;
    font-weight: 700;
}

.result-text {
    font-size: 11px;
    margin-top: 3px;
    color: #71809c;
}

/* ---------------- METRICS ---------------- */

.metric-blue {
    background: #f1f5ff;
    border: 1px solid #dce5ff;
    padding: 14px;
    border-radius: 8px;
}

.metric-red {
    background: #fff1f4;
    border: 1px solid #f5d8df;
    padding: 14px;
    border-radius: 8px;
}

.metric-label {
    font-size: 11px;
    color: #5c6b87;
}

.metric-value {
    font-size: 24px;
    font-weight: 700;
    color: #263b6b;
}

/* ---------------- BUTTON ---------------- */

.stButton > button {
    width: 100%;
    border-radius: 7px;
    border: none;
    background: linear-gradient(90deg, #4169e1, #4c6ff0);
    color: white;
    font-weight: 600;
    padding: 11px;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #3458c8, #405fd4);
    color: white;
}

/* ---------------- FOOTER ---------------- */

.footer {
    background: white;
    border: 2px solid #e1e6ef;
    border-radius: 15px;
    padding: 10px 15px;
    font-size: 10px;
    color: #6e7b94;
    margin-top: 8px;
}

/* ---------------- INPUT LABEL ---------------- */

label {
    color: #263b62 !important;
    font-weight: 600 !important;
    font-size: 12px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TOP HEADER
# =========================================================

st.markdown("""
<div class="top-header">
    <div class="logo-area">
        <div class="heart-logo">♥</div>
        <div>
            <div class="header-title">Heart Disease Prediction</div>
            <div class="header-subtitle">
                AI-Powered Health Analysis &nbsp; • &nbsp;
                Better Insights &nbsp; • &nbsp;
                Healthier Tomorrow
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">❤️ Heart Disease AI</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🔍 Predict",
            "📊 Visualizations",
            "🏆 Model Performance",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">
            ❤️ Early Detection Saves Lives
        </div>
        <div class="sidebar-card-text">
            Use data-driven machine learning to estimate
            the risk of heart disease.
            <br><br>
            Make an informed decision with AI-assisted
            prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PREDICTION PAGE
# =========================================================

if page == "🔍 Predict":

    left, right = st.columns([1.05, 1.25], gap="medium")

    # =====================================================
    # LEFT SIDE - PATIENT INFORMATION
    # =====================================================

    with left:

        st.markdown("""
        <div class="card">
            <div class="card-title">👤 Patient Information</div>
            <div class="card-subtitle">
                Please enter the following details to predict
                the risk of heart disease.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            patient_name = st.text_input(
            "Patient Name",
            placeholder="Enter patient's name"
        )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=40
            )

            education = st.number_input(
                "Education Level",
                min_value=1,
                max_value=4,
                value=2
            )

            currentSmoker = st.selectbox(
                "Current Smoker",
                [0, 1],
                format_func=lambda x:
                    "No" if x == 0 else "Yes"
            )

            cigsPerDay = st.number_input(
                "Cigarettes Per Day",
                min_value=0,
                max_value=100,
                value=0
            )

            BPMeds = st.selectbox(
                "Blood Pressure Medication",
                [0, 1],
                format_func=lambda x:
                    "No" if x == 0 else "Yes"
            )

            prevalentStroke = st.selectbox(
                "Previous Stroke",
                [0, 1],
                format_func=lambda x:
                    "No" if x == 0 else "Yes"
            )

            prevalentHyp = st.selectbox(
                "Hypertension",
                [0, 1],
                format_func=lambda x:
                    "No" if x == 0 else "Yes"
            )


        with col2:

            male = st.selectbox(
                "Sex",
                ["Female", "Male"]
            )

            totChol = st.number_input(
                "Total Cholesterol (mg/dL)",
                min_value=50.0,
                max_value=500.0,
                value=200.0
            )

            sysBP = st.number_input(
                "Systolic Blood Pressure",
                min_value=50.0,
                max_value=300.0,
                value=120.0
            )

            diaBP = st.number_input(
                "Diastolic Blood Pressure",
                min_value=30.0,
                max_value=200.0,
                value=80.0
            )

            BMI = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=70.0,
                value=25.0
            )

            heartRate = st.number_input(
                "Heart Rate",
                min_value=30.0,
                max_value=200.0,
                value=70.0
            )

            glucose = st.number_input(
                "Glucose",
                min_value=40.0,
                max_value=500.0,
                value=100.0
            )

            diabetes = st.selectbox(
                "Diabetes",
                [0, 1],
                format_func=lambda x:
                "No" if x == 0 else "Yes"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        predict_button = st.button(
            "❤️  Predict Heart Disease",
            use_container_width=True
        )

    # =====================================================
    # RIGHT SIDE
    # =====================================================

    with right:

        if predict_button:

            gender = 1 if male == "Male" else 0

            input_data = pd.DataFrame({
                "male": [gender],
                "age": [age],
                "education": [education],
                "currentSmoker": [currentSmoker],
                "cigsPerDay": [cigsPerDay],
                "BPMeds": [BPMeds],
                "prevalentStroke": [prevalentStroke],
                "prevalentHyp": [prevalentHyp],
                "diabetes": [diabetes],
                "totChol": [totChol],
                "sysBP": [sysBP],
                "diaBP": [diaBP],
                "BMI": [BMI],
                "heartRate": [heartRate],
                "glucose": [glucose]
            })

            # Prediction
            prediction = model.predict(input_data)[0]

            # Probability
            probabilities = model.predict_proba(input_data)[0]

            no_disease_probability = probabilities[0] * 100
            disease_probability = probabilities[1] * 100

            # =================================================
            # RESULT
            # =================================================

            if prediction == 1:

                st.markdown(f"""
                <div class="result-danger">
                    <div class="result-title">
                        ⚠️ Prediction: Higher Risk of Heart Disease
                    </div>
                    <div class="result-text">
                        The model predicts a higher likelihood of
                        heart disease based on the provided information.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="result-success">
                    <div class="result-title">
                        ✅ Prediction: Lower Risk of Heart Disease
                    </div>
                    <div class="result-text">
                        The model predicts a lower likelihood of
                        heart disease based on the provided information.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # =================================================
            # PROBABILITY CARDS
            # =================================================

            c1, c2 = st.columns(2)

            with c1:

                st.markdown(f"""
                <div class="metric-blue">
                    <div class="metric-label">
                        🛡️ Confidence (No Disease)
                    </div>
                    <div class="metric-value">
                        {no_disease_probability:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c2:

                st.markdown(f"""
                <div class="metric-red">
                    <div class="metric-label">
                        ⚠️ Confidence (Disease)
                    </div>
                    <div class="metric-value">
                        {disease_probability:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

                        # =================================================
            # MODEL INFORMATION
            # =================================================

            st.subheader("🧠 Model Information")

            model_name = type(model).__name__

            if hasattr(model, "named_steps"):
                final_model = list(model.named_steps.values())[-1]
                model_name = type(final_model).__name__
            else:
                final_model = model

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Best Model Used**")
                st.write(model_name)

            with col2:
                st.markdown("**Algorithm**")
                st.write("Ensemble Learning")

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("**Dataset**")
                st.write("Framingham Heart Disease")

            with col4:
                st.markdown("**Problem Type**")
                st.write("Binary Classification")
            # =================================================
            # FEATURE IMPORTANCE
            # =================================================

            st.markdown("""
            <div class="card">
                <div class="card-title">
                    📊 Feature Importance
                </div>
            """, unsafe_allow_html=True)

            feature_names = [
                "Gender",
                "Age",
                "Education",
                "Current Smoker",
                "Cigarettes/Day",
                "BP Medication",
                "Previous Stroke",
                "Hypertension",
                "Diabetes",
                "Total Cholesterol",
                "Systolic BP",
                "Diastolic BP",
                "BMI",
                "Heart Rate",
                "Glucose"
            ]

            importance = None

            if hasattr(final_model, "feature_importances_"):
                importance = final_model.feature_importances_

            if importance is not None:

                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importance
                }).sort_values(
                    "Importance",
                    ascending=False
                ).head(8)

                fig, ax = plt.subplots(figsize=(6, 3.5))

                ax.barh(
                    importance_df["Feature"][::-1],
                    importance_df["Importance"][::-1]
                )

                ax.set_xlabel("Importance")
                ax.set_title("Top Important Features")

                plt.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

            else:

                st.info(
                    "Feature importance is not available for this model."
                )

            st.markdown("</div>", unsafe_allow_html=True)

            # =================================================
            # HEART RATE TREND
            # =================================================
            st.markdown("""
            <div class = "card">
                <div class="card-title">
                   ❤️ Heart Rate Trend (Sample Data)
                </div>
            </div>
            """,unsafe_allow_html=True
            )

            # Sample heart-rate data
            heart_rate_data = [
                72, 75, 73, 78, 81,
                79, 85, 88, 84, 90,
                92, 87, 95, 98, 91,
                96, 100, 103, 98, 105,
                108, 112, 109, 115, 118,
                120, 117, 123, 126, 130
            ]

            fig, ax = plt.subplots(figsize=(8, 3.5))

            ax.plot(
                range(len(heart_rate_data)),
                heart_rate_data,
                linewidth = 2,
                marker = "o",
                markersize = 3
            )

            ax.set_xlabel("Time(sample)")
            ax.set_ylabel("Heart Rate")
            ax.set_title("Heart Rate Trend")

            ax.grid(
                alpha=0.2,
                linestyle = "--"
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        else:

            # Initial screen before prediction

            st.markdown("""
            <div class="card">
                <div class="card-title"> 🧠 Prediction Result </div>
                <div class = "card-subtitle">
                    Enter the patient's information and click
                    <b>Predict Heart Disease</b> to get
                    the machine learning prediction.
                </div>
            </div>
            """, unsafe_allow_html=True
            )

            st.markdown(
                """
                <div style = "
                background:white;
                border:1px solid #e2e7f0;
                border-radius:10px;
                padding:55px 20px;
                margin-top:10px;
            ">
                <div style="font-size:50px;">❤️</div>
                <h2 style="color:#263b62;">
                    Ready for Prediction
                </h2>
                <p style="color:#71809c;">
                   Enter the patient's information and click
                   <b>Predict Heart Disease</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
            )


# =========================================================
# VISUALIZATIONS PAGE
# =========================================================

elif page == "📊 Visualizations":

    st.title("📊 Data Visualizations")

    st.info(
        "This section can be used to display dataset-level "
        "visualizations such as distributions, correlations "
        "and target analysis."
    )

    csv_path = BASE_DIR / "heartpred.csv"

    if csv_path.exists():

        df = pd.read_csv(csv_path)

        st.subheader("Dataset Overview")

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Records", len(df))
        c2.metric("Total Features", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.subheader("Age Distribution")

        if "age" in df.columns:

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.hist(df["age"], bins=25)

            ax.set_xlabel("Age")
            ax.set_ylabel("Number of Patients")
            ax.set_title("Age Distribution")

            st.pyplot(fig)

            plt.close(fig)

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )


# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================

elif page == "🏆 Model Performance":

    st.title("🏆 Model Performance")

    st.markdown("""
    <div class="card">
        <div class="card-title">
            Machine Learning Model
        </div>
        <p>
            The application uses the trained best-performing
            classification model saved in <b>best_model.pkl</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("### Model Details")

    model_name = type(model).__name__

    if hasattr(model, "named_steps"):
        final_model = list(model.named_steps.values())[-1]
        model_name = type(final_model).__name__

    c1, c2, c3 = st.columns(3)

    c1.metric("Algorithm", model_name)
    c2.metric("Problem", "Binary Classification")
    c3.metric("Output", "0 / 1")


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    <div class="card"  style="background-color:#FFF5EE;">

    <h3 style="color:#263b62;">
    ❤️ Heart Disease Prediction
    </h3>

    <p>
    This project demonstrates the use of Machine Learning
    for predicting the risk of heart disease from patient
    health-related information.
    </p>

    <h4>Technologies Used</h4>

    <ul>
        <li>Python</li>
        <li>Pandas</li>
        <li>NumPy</li>
        <li>Scikit-learn</li>
        <li>Random Forest / Machine Learning</li>
        <li>Streamlit</li>
        <li>Joblib</li>
        <li>Matplotlib</li>
    </ul>

    <h4>Project Workflow</h4>

    <p>
    Data → Preprocessing → Model Training →
    Model Evaluation → Best Model → Streamlit Deployment
    </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    ❤️ This project demonstrates Machine Learning based
    heart disease risk prediction using the Framingham dataset.
    <span style="float:right;">
        Deployed by <b>Niraj Gupta</b> &nbsp; • &nbsp;
        Built with Streamlit
    </span>
</div>
""", unsafe_allow_html=True)