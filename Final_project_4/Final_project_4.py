import streamlit as st
import joblib
import numpy as np
import time

model = joblib.load(r"C:\Users\SYAMNARAYANAN\OneDrive\Desktop\visual code\Guvi_Project_3\final_project_4 drop_out.pkl")


st.title("🎓 Student Dropout Prediction App")

st.write("""
Welcome! Use the sidebar to navigate through the app:
- 🧾 **Personal Info**: Input student background  
- 🔍 **Prediction**: Make a prediction based on inputs  
- ℹ️ **About**: App info and credits
""")


st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)),
                    url("https://static.vecteezy.com/system/resources/previews/049/193/239/original/business-people-walking-up-the-stairs-to-the-graduation-cap-vector.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #f9f9f9;
        font-family: 'Segoe UI';
    }
    </style>
    """,True)


tab1, tab2 = st.tabs(["Personal Info", "Academic Info"])

with tab1:
    admission_taken = st.selectbox("Admission Taken?", ["No", "Yes"])
    mental_health_issues = st.selectbox("Mental Health Issues", ["No", "Yes"])
    peer_pressure_level = st.selectbox("peer pressure level",["High", "Mid","Low"])
    parent_education = st.selectbox("Parent Education Level", ["PG", "Graduate", "12th","Upto 10th"])
    Family_Income = st.selectbox("Family Income",["High", "Mid","Low"])



with tab2:
    jee_main_score = st.number_input("JEE Main Score(%)", 0, 100, 70)
    jee_advanced_score = st.number_input("JEE Advanced Score(%)", 0, 100, 50)
    mock_test_score_avg = st.number_input("Mock Test Avg (%)", 0, 100, 60)
    class_12_percent = st.number_input("Class 12 Percentage", 0, 100, 70)
    daily_study_hours = st.slider("Daily Study Hours", 0.0, 12.0, 2.0)


if st.button("🔍 Predict Dropout Risk"):
    admission_taken_num = 1 if admission_taken == "Yes" else 0
    parent_education_num = {'Upto 10th': 1, 'PG': 4, '12th': 2, 'Graduate': 3}[parent_education]
    mental_health_issues_num = 1 if mental_health_issues == "Yes" else 0
    peer_pressure_level_num = {'High': 2, 'Mid': 1, 'Low': 0}[peer_pressure_level]
    Family_Income_num = {'High': 2, 'Mid': 1, 'Low': 0}[Family_Income]

    features = np.array([[Family_Income_num, admission_taken_num, daily_study_hours,
                          peer_pressure_level_num, jee_main_score, jee_advanced_score,
                          mock_test_score_avg, class_12_percent,
                          parent_education_num, mental_health_issues_num]])

    prediction = model.predict(features)[0]

    placeholder = st.empty() 

    st.markdown("---")
    with placeholder:
        if prediction == 1:
            st.error("⚠️ High Risk: Likely to Drop Out")
        else:
            st.success("Low Risk: Likely to Continue")

    time.sleep(2)
    placeholder.empty()


st.title("📘 About This App")
st.markdown("---")

st.markdown("""
### 🎯 Purpose
This app helps predict the **risk of a student dropping out** using various academic and personal parameters. By assessing early warning signs, institutions can offer **timely interventions**.

---

### 🔍 How It Works
- You provide student-related inputs across two main tabs:
  1. **Personal Info** – like family income, parental education, mental health status, etc.
  2. **Academic Info** – like JEE scores, mock test performance, and study hours.

- A **Random Forest Machine Learning model** processes the data and gives:
  - 🟢 **Low Risk** – likely to continue education.
  - 🔴 **High Risk** – may require intervention.

---

### 🧠 Model Used
- Trained using **Random Forest Classifier**.
- Enhanced with **SMOTE** to balance class distribution.
- Features like:
  - Family income
  - Peer pressure
  - Study hours
  - Mental health
  - JEE/Main scores

""")


