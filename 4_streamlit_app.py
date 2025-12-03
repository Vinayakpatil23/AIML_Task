"""
Heart Disease Classification - Streamlit Frontend
Interactive UI for testing predictions
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stTitle {
        color: #e74c3c;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #e74c3c;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== CONSTANTS ====================
API_URL = "http://localhost:8000"  # Change if API runs on different port

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("⚙️ Configuration")
    
    api_status = st.empty()
    
    # Check API connectivity
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            api_status.success("✓ API Connected")
        else:
            api_status.error("✗ API Error")
    except:
        api_status.error("✗ API Offline")
    
    st.divider()
    
    mode = st.radio(
        "Select Mode:",
        ["Single Prediction", "Batch Upload", "Sample Test"],
        index=0
    )
    
    st.divider()
    
    with st.expander("ℹ️ About"):
        st.write("""
        **Heart Disease Prediction System**
        
        This application predicts the risk of heart disease based on patient health indicators.
        
        - Uses ML classification model
        - Provides confidence scores
        - Risk assessment (Low/Moderate/High)
        """)

# ==================== MAIN TITLE ====================
st.title("❤️ Heart Disease Prediction System")
st.markdown("Predict heart disease risk using AI/ML")

# ==================== MODE 1: SINGLE PREDICTION ====================
if mode == "Single Prediction":
    st.header("Single Patient Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        age = st.slider("Age", 29, 77, 50)
        sex = st.radio("Sex", ["Female (0)", "Male (1)"], index=1)
        sex_value = 0 if "Female" in sex else 1
    
    with col2:
        st.subheader("Chest Pain")
        cp = st.selectbox("Chest Pain Type", 
            [0, 1, 2, 3], 
            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"][x])
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Blood Pressure")
        trestbps = st.slider("Resting BP (mmHg)", 90, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 126, 564, 240)
    
    with col4:
        st.subheader("Blood Sugar & ECG")
        fbs = st.radio("Fasting Blood Sugar > 120?", ["No (0)", "Yes (1)"], index=0)
        fbs_value = 0 if "No" in fbs else 1
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("Heart Activity")
        thalach = st.slider("Max Heart Rate", 60, 202, 130)
        exang = st.radio("Exercise Induced Angina?", ["No (0)", "Yes (1)"], index=0)
        exang_value = 0 if "No" in exang else 1
    
    with col6:
        st.subheader("Other Indicators")
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
        slope = st.selectbox("ST Segment Slope", [0, 1, 2])
    
    col7, col8 = st.columns(2)
    
    with col7:
        ca = st.slider("Major Vessels (0-4)", 0, 4, 0)
    
    with col8:
        thal = st.selectbox("Thalassemia Type", 
            [0, 1, 2, 3],
            format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x] if x < 3 else "Unknown")
    
    # Predict button
    if st.button("🔍 Predict", use_container_width=True, type="primary"):
        try:
            payload = {
                "age": age,
                "sex": sex_value,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs_value,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang_value,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal
            }
            
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                st.divider()
                st.subheader("📊 Prediction Result")
                
                col_result1, col_result2, col_result3 = st.columns(3)
                
                with col_result1:
                    if result['prediction'] == 0:
                        st.success("✓ No Disease Detected")
                    else:
                        st.error("⚠️ Disease Risk Detected")
                
                with col_result2:
                    st.metric("Confidence", f"{result['confidence']:.2%}")
                
                with col_result3:
                    risk_color = "🟢" if result['risk_level'] == "low" else "🟡" if result['risk_level'] == "moderate" else "🔴"
                    st.metric("Risk Level", f"{risk_color} {result['risk_level'].upper()}")
                
                st.info(f"**Model Used**: {result['model']}")
            else:
                st.error(f"API Error: {response.status_code}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ==================== MODE 2: BATCH UPLOAD ====================
elif mode == "Batch Upload":
    st.header("Batch Prediction Upload")
    
    st.write("Upload a CSV file with patient data. Required columns:")
    st.code("""age, sex, cp, trestbps, chol, fbs, restecg, 
thalach, exang, oldpeak, slope, ca, thal""")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded {len(df)} patients")
        st.dataframe(df.head())
        
        if st.button("🔍 Predict All", use_container_width=True, type="primary"):
            try:
                # Convert to list of dicts
                patients = df.to_dict('records')
                
                payload = {"patients": patients}
                response = requests.post(f"{API_URL}/predict-batch", json=payload)
                
                if response.status_code == 200:
                    results = response.json()
                    
                    st.success(f"✓ Predictions for {results['total_patients']} patients")
                    
                    # Display results
                    results_data = []
                    for result in results['results']:
                        results_data.append({
                            "Age": result['patient_data']['age'],
                            "Prediction": "Disease" if result['prediction'] == 1 else "No Disease",
                            "Confidence": f"{result['confidence']:.2%}",
                            "Risk Level": result['risk_level'].upper()
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"API Error: {response.status_code}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== MODE 3: SAMPLE TEST ====================
elif mode == "Sample Test":
    st.header("Test with Sample Data")
    
    # Sample patients
    samples = {
        "Healthy Person": {
            "age": 40, "sex": 0, "cp": 0, "trestbps": 120, "chol": 200,
            "fbs": 0, "restecg": 0, "thalach": 120, "exang": 0,
            "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 1
        },
        "At-Risk Patient": {
            "age": 60, "sex": 1, "cp": 3, "trestbps": 150, "chol": 300,
            "fbs": 1, "restecg": 2, "thalach": 100, "exang": 1,
            "oldpeak": 3.5, "slope": 0, "ca": 3, "thal": 2
        },
        "High-Risk Patient": {
            "age": 70, "sex": 1, "cp": 1, "trestbps": 160, "chol": 350,
            "fbs": 1, "restecg": 2, "thalach": 90, "exang": 1,
            "oldpeak": 4.0, "slope": 0, "ca": 4, "thal": 3
        }
    }
    
    for name, data in samples.items():
        with st.expander(f"📋 {name}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(data)
            
            with col2:
                if st.button(f"Predict", key=name):
                    try:
                        response = requests.post(f"{API_URL}/predict", json=data)
                        if response.status_code == 200:
                            result = response.json()
                            st.json({
                                "Prediction": "Disease" if result['prediction'] == 1 else "No Disease",
                                "Confidence": f"{result['confidence']:.2%}",
                                "Risk Level": result['risk_level'].upper()
                            })
                        else:
                            st.error("API Error")
                    except Exception as e:
                        st.error(str(e))

# ==================== FOOTER ====================
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Heart Disease Prediction System | ML Classification Model | Powered by FastAPI & Streamlit
    </div>
    """, unsafe_allow_html=True)
