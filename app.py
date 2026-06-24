import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Set page layout configuration
st.set_page_config(page_title="Healthcare ML", page_icon="🏥", layout="wide")

# Custom CSS styling for a clean medical UI dashboard
st.markdown("""
<style>
    .main-header {font-size: 42px; font-weight: bold; color: #1f77b4; text-align: center;}
    .prediction-box {padding: 20px; border-radius: 10px; margin: 10px 0;}
    .high-risk {background-color: #ffebee; border-left: 5px solid #f44336;}
    .medium-risk {background-color: #fff3e0; border-left: 5px solid #ff9800;}
    .low-risk {background-color: #e8f5e9; border-left: 5px solid #4caf50;}
</style>
""", unsafe_allow_html=True)

# Optimized caching to load ML model files once and prevent performance lags
@st.cache_resource
def load_models():
    try:
        # Expects pickled pipeline files to sit in the same root folder
        diabetes_model = joblib.load('diabetes_model.pkl')
        heart_model = joblib.load('heart_disease_model.pkl')
        scaler_diabetes = joblib.load('scaler_diabetes.pkl')
        scaler_heart = joblib.load('scaler_heart.pkl')
        return diabetes_model, heart_model, scaler_diabetes, scaler_heart, None
    except Exception as e:
        return None, None, None, None, str(e)

# Render Application Headers
st.markdown('<div class="main-header">🏥 Healthcare Predictive Analytics</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #666; margin-bottom: 30px;">MIT Anna University | ML-Powered Disease Risk Assessment</div>', unsafe_allow_html=True)

# Sidebar Navigation Panel
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Disease", ["🩺 Diabetes", "❤️ Heart Disease", "📊 About"])

# Initialize and verify models load correctly
diabetes_model, heart_model, scaler_diabetes, scaler_heart, error = load_models()

if error:
    st.error(f"⚠️ Error loading models: {error}")
    st.info("Ensure that 'diabetes_model.pkl', 'heart_disease_model.pkl', 'scaler_diabetes.pkl', and 'scaler_heart.pkl' are uploaded into the same root folder on your repository.")
    st.stop()

# ==========================================
# PAGE 1: DIABETES ASSESSMENT
# ==========================================
if page == "🩺 Diabetes":
    st.header("Diabetes Risk Assessment")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
        bp = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
        skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, 0.1)
        dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5, 0.01)
        age = st.number_input("Age", 1, 120, 30)

    if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
        features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        features_scaled = scaler_diabetes.transform(features)
        prediction = diabetes_model.predict(features_scaled)[0]
        probability = diabetes_model.predict_proba(features_scaled)[0][1]

        risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        risk_class = "high-risk" if probability > 0.7 else "medium-risk" if probability > 0.4 else "low-risk"
        color = "#f44336" if probability > 0.7 else "#ff9800" if probability > 0.4 else "#4caf50"

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Diagnosis", "Diabetic" if prediction == 1 else "Non-Diabetic")
        col2.metric("Risk Probability", f"{probability:.1%}")
        col3.metric("Risk Level", risk_level)

        # Plotly risk gauge display
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=probability * 100,
            title={'text': "Risk Score"},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': color},
                   'steps': [{'range': [0, 40], 'color': "#e8f5e9"},
                            {'range': [40, 70], 'color': "#fff3e0"},
                            {'range': [70, 100], 'color': "#ffebee"}]}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f'<div class="prediction-box {risk_class}"><h3>📋 Recommendations</h3>', unsafe_allow_html=True)
        if risk_level == "High":
            st.markdown("""
            - ⚠️ **Immediate Action Required**: Consult an endocrinologist
            - 🍎 Follow a strict diabetic diet plan
            - 💊 Medication may be necessary
            - 🏃 Regular exercise (30 min/day)
            - 📊 Monitor blood glucose daily
            """)
        elif risk_level == "Medium":
            st.markdown("""
            - ⚡ **Preventive Measures Recommended**
            - 🥗 Maintain a balanced diet, reduce sugar intake
            - 🏃 Exercise regularly (150 min/week)
            - ⚖️ Maintain healthy weight
            - 🔬 Regular check-ups every 6 months
            """)
        else:
            st.markdown("""
            - ✅ **Good Health Status**
            - 🥗 Continue healthy eating habits
            - 🏃 Stay physically active
            - 🔬 Annual health screening recommended
            - 📚 Stay informed about diabetes prevention
            """)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 2: HEART DISEASE ASSESSMENT
# ==========================================
elif page == "❤️ Heart Disease":
    st.header("Heart Disease Risk Assessment")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
        cp = st.selectbox("Chest Pain", [("Typical Angina", 0), ("Atypical Angina", 1),
                                         ("Non-anginal", 2), ("Asymptomatic", 3)], format_func=lambda x: x[0])
        trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
        chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    with col2:
        fbs = st.selectbox("Fasting BS > 120", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        restecg = st.selectbox("Resting ECG", [("Normal", 0), ("ST-T Abnormality", 1),
                                                ("LV Hypertrophy", 2)], format_func=lambda x: x[0])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Angina", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
    with col3:
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, 0.1)
        slope = st.selectbox("ST Slope", [("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)], format_func=lambda x: x[0])
        ca = st.number_input("Major Vessels (0-3)", 0, 3, 0)
        thal = st.selectbox("Thalassemia", [("Normal", 1), ("Fixed", 2), ("Reversible", 3)], format_func=lambda x: x[0])

    if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):
        features = np.array([[age, sex[1], cp[1], trestbps, chol, fbs[1], restecg[1],
                             thalach, exang[1], oldpeak, slope[1], ca, thal[1]]])
        features_scaled = scaler_heart.transform(features)
        prediction = heart_model.predict(features_scaled)[0]
        probability = heart_model.predict_proba(features_scaled)[0][1]

        risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        risk_class = "high-risk" if probability > 0.7 else "medium-risk" if probability > 0.4 else "low-risk"
        color = "#f44336" if probability > 0.7 else "#ff9800" if probability > 0.4 else "#4caf50"

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Diagnosis", "Heart Disease" if prediction == 1 else "No Disease")
        col2.metric("Risk Probability", f"{probability:.1%}")
        col3.metric("Risk Level", risk_level)

        # Plotly risk gauge display
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=probability * 100,
            title={'text': "Risk Score"},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': color},
                   'steps': [{'range': [0, 40], 'color': "#e8f5e9"},
                            {'range': [40, 70], 'color': "#fff3e0"},
                            {'range': [70, 100], 'color': "#ffebee"}]}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f'<div class="prediction-box {risk_class}"><h3>📋 Recommendations</h3>', unsafe_allow_html=True)
        if risk_level == "High":
            st.markdown("""
            - ⚠️ **Urgent Medical Attention Required**: See a cardiologist immediately
            - 💊 May require medication or intervention
            - 🏥 Comprehensive cardiac evaluation needed
            - 🚭 Stop smoking immediately if applicable
            - 🧂 Strict low-sodium, heart-healthy diet
            - 🏃 Supervised exercise program
            """)
        elif risk_level == "Medium":
            st.markdown("""
            - ⚡ **Preventive Care Recommended**
            - 🏥 Schedule cardiac screening
            - 🥗 Heart-healthy diet (low fat, low sodium)
            - 🏃 Regular moderate exercise
            - 💊 Monitor blood pressure and cholesterol
            - 😴 Manage stress and get adequate sleep
            """)
        else:
            st.markdown("""
            - ✅ **Good Cardiovascular Health**
            - 🥗 Continue heart-healthy lifestyle
            - 🏃 Maintain regular physical activity
            - 🔬 Annual cardiovascular check-ups
            - 📚 Stay informed about heart health
            """)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 3: PROJECT BACKGROUND INFO
# ==========================================
else:
    st.header("About This System")
    st.markdown("""
    ### 🎯 Project Overview
    Predictive Healthcare Analytics using Machine Learning for diabetes and heart disease detection.

    ### 🔬 ML Models (Scikit-learn)
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - SVM
    - Neural Network
    - Naive Bayes
    - K-Means Clustering

    ### 📊 Datasets
    - **PIMA Diabetes**: 768 patients, 8 features
    - **UCI Heart Disease**: 303 patients, 13 features

    ### 👥 Developed By
    **Sripathy V** (2023506113) & **Dinitha Poola** (2023506127)
    Department of Information Technology, MIT Anna University

    ### ⚠️ Disclaimer
    For educational purposes only. Consult healthcare professionals for medical decisions.
    """)

st.markdown("---")
st.markdown('<div style="text-align: center; color: #666;">🏥 Healthcare ML System v1.0 | MIT Anna University 2024-25</div>', unsafe_allow_html=True)