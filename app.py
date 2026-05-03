import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# 🚨 Fix for Hugging Face Spaces
if __name__ == "__main__":
    os.system("streamlit run app.py --server.port 7860 --server.address 0.0.0.0")

# ---------------- LOAD ----------------
@st.cache_resource
def load_all():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_all()

# ---------------- UI ----------------
st.set_page_config(page_title="Dry Eye Predictor", layout="wide")

st.markdown("""
<div style='background:linear-gradient(90deg,#0D1B4B,#1565C0);
padding:25px;border-radius:12px;margin-bottom:20px'>
<h1 style='color:white;text-align:center;'>👁️ Dry Eye Disease Predictor</h1>
<p style='color:#BBDEFB;text-align:center;margin-top:5px;'>
AI-powered screening using lifestyle & health indicators
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("## 📋 User Input")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Basic Info")
    age = st.slider("Age", 18, 45, 25)
    sleep = st.slider("Sleep Hours", 4.0, 10.0, 7.0)
    stress = st.slider("Stress Level", 1, 5, 3)

with col2:
    st.markdown("### 🖥️ Lifestyle")
    screen = st.slider("Screen Time (hrs)", 1.0, 10.0, 5.0)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    eye_strain = st.selectbox("Eye Strain", ["No", "Yes"])

# ---------------- PREDICT ----------------
if st.button("Predict"):

    try:
        user_input = {
            "Age": age,
            "Sleep duration": sleep,
            "Stress level": stress,
            "Average screen time": screen,
            "Smoking": "Y" if smoking == "Yes" else "N",
            "Discomfort Eye-strain": "Y" if eye_strain == "Yes" else "N",
        }

        # Build full feature vector
        row = {}
        for col in feature_names:
            if col in user_input:
                val = user_input[col]
                if val == "Y":
                    row[col] = 1
                elif val == "N":
                    row[col] = 0
                else:
                    row[col] = val
            else:
                row[col] = 0

        X = np.array([list(row.values())], dtype=float)

        # Scale + Predict
        X = scaler.transform(X)
        prob = model.predict_proba(X)[0][1] * 100

        # ---------------- RESULT ----------------
        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        st.progress(int(prob) / 100)

        if prob > 60:
            color = "#F44336"
            label = "High Risk"
        elif prob > 30:
            color = "#FF9800"
            label = "Medium Risk"
        else:
            color = "#4CAF50"
            label = "Low Risk"

        st.markdown(f"""
        <div style='background:white;padding:20px;border-radius:12px;
        box-shadow:0 2px 10px rgba(0,0,0,0.1);text-align:center;'>

        <h2 style='color:{color};'>{label}</h2>
        <h1 style='color:{color};'>{prob:.2f}%</h1>

        </div>
        """, unsafe_allow_html=True)

        # ---------------- RECOMMENDATIONS ----------------
        st.markdown("## 💡 Personalized Recommendations")

        if prob > 60:
            st.error("🚨 High Risk — Immediate Action Required")
            st.write("• Reduce screen time drastically")
            st.write("• Improve sleep (7–9 hours)")
            st.write("• Avoid screens before bed")
            st.write("• Use lubricating eye drops")
            st.write("• Consult eye specialist")

        elif prob > 30:
            st.warning("⚠️ Moderate Risk — Needs Attention")
            st.write("• Follow 20-20-20 rule")
            st.write("• Reduce stress")
            st.write("• Blink frequently")
            st.write("• Stay hydrated")

        else:
            st.success("✅ Low Risk — Maintain Healthy Habits")
            st.write("• Maintain current routine")
            st.write("• Take screen breaks")
            st.write("• Keep balanced lifestyle")

    except Exception as e:
        st.error(f"Error: {e}")