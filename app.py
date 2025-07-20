import streamlit as st
import pandas as pd
import joblib
import os
from streamlit_extras.let_it_rain import rain

# --- Page Configuration ---
st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

# --- Load Assets ---
def load_asset(file_name):
    try:
        return joblib.load(file_name)
    except FileNotFoundError:
        return None

model = load_asset('regression_model.pkl')
metrics = load_asset('performance_metrics.pkl')

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file '{file_name}' not found.")

load_css("style.css")

# --- Check for required files ---
if not all([model, metrics, os.path.exists('feature_importance.png'), os.path.exists('scatter_plot.png'), os.path.exists('line_plot.png')]):
    st.error("❌ Critical files are missing. Please run `train_model.py` to generate them.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.title("Employee Salary Predictor")
    st.markdown("""
    An AI-powered salary estimation tool based on:
    - Age
    - Gender
    - Education Level
    - Job Title
    - Years of Experience
    """)
    
    st.header("Model Performance")
    st.markdown("Selected Model: **Linear Regression**")
    
    for label, value in metrics.items():
        st.markdown(f"""
        <div class="performance-metric">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# --- Main Content ---
st.header("SalarySense Tool")

# --- Input Fields ---
col1, col2 = st.columns(2, gap="large")

with col1:
    age = st.slider("Age", 18, 65, 30)
    gender = st.radio("Gender", ["Male", "Female"])
    education_level = st.radio("Education Level", ["Bachelor's", "Master's", "PhD"])

with col2:
    job_title = st.selectbox("Job Title", ["Software Engineer", "Data Scientist", "Project Manager", "Business Analyst", "DevOps Engineer", "UI/UX Designer"])
    years_of_experience = st.slider("Years of Experience", 0, 40, 5)

# --- Prediction Logic ---
if st.button("Predict Salary"):
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education_level,
        'Job Title': job_title,
        'Years of Experience': years_of_experience
    }])
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"Predicted Salary: **${prediction:,.2f}** per year")
    
    st.balloons()
    
    # 💵 Cash Rain animation (stops after 2 seconds)
    rain(
        emoji="💵",
        font_size=54,
        falling_speed=5,
        animation_length=2  # changed from 5 to 2 seconds
    )

# --- Result Visualizations ---
st.markdown("---")
st.header("Model Insights")

with st.expander("Feature Importance"):
    st.image('feature_importance.png')

with st.expander("Scatter Plot: Actual vs Predicted Salary"):
    st.image('scatter_plot.png')

with st.expander("Line Plot: Residuals (Error Distribution)"):
    st.image('line_plot.png')
