import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

model = joblib.load(r"C:\Users\SYAMNARAYANAN\OneDrive\Desktop\visual code\Guvi_Project_3\final_project_pass_fail_1.pkl")


st.title("Predict Students Pass or Fail")
st.info("This app predicts whether a student will **Pass or Fail** based on their input values.")


st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                    url("https://images.unsplash.com/photo-1529156069898-49953e39b3ac");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #f9f9f9;
        font-family: 'Segoe UI';
    }

    h1 {
        font-family: 'Georgia';
        font-size: 3em;
        text-align: center;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.8);
        color: #fff;
    }

    label {
        color: #fff !important;
        font-weight: 600;
    }

    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #2E8B57);
        color: white;
        font-weight: bold;
        border-radius: 15px;
        padding: 12px 30px;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.4s ease-in-out;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(270deg, #ff416c, #ff4b2b, #ff6a00);
        background-size: 600% 600%;
        color: #ffffe0;
        animation: gradientShift 4s ease infinite;
        transform: scale(1.05);
    }

    .custom-box {
        padding: 1em;
        border-radius: 10px;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
        animation: fadeOut 3s ease-in-out forwards;
    }

    .pass-box {
        background-color: #d4edda;
        color: #155724;
    }

    .fail-box {
        background-color: #f8d7da;
        color: #721c24;
    }

    @keyframes fadeOut {
        0% {opacity: 1;}
        80% {opacity: 1;}
        100% {opacity: 0;}
    }

    </style>
""", unsafe_allow_html=True)



tab_1, tab_2 = st.tabs(["🎯 Prediction", "ℹ️ About"])


with tab_1:
    with st.expander("📄 Dataset"):
        dataset = pd.read_csv(r"C:\Users\SYAMNARAYANAN\OneDrive\Desktop\visual code\Guvi_Project_3\student_dataset_1.csv")
        st.dataframe(dataset)

    with st.expander("🔍 Input Features Only"):
        view_dataset = dataset[['hours_studied', 'previous_grade', 'attendance', 'sleep_hours', 'play_hours',
                                'daily_travel_time', 'likes_subject', 'health', 'study_group']]
        st.dataframe(view_dataset)

    with st.sidebar:
        st.image("https://www.kevinmd.com/wp-content/uploads/shutterstock_213089176-3.jpg", use_container_width=True)

        st.header("🔢 Input Features")
        Hours_Studied = st.slider("Hours Studied", 1, 10, 2)
        Previous_Grade = st.number_input("Previous Grade", 1, 100, 50)
        Attendance = st.slider("Attendance", 1, 100, 70)
        Sleep_hours = st.slider("Sleep hours", 2, 10, 5)
        Play_hours = st.slider("Play hours", 2, 5, 2)
        Health = st.slider("Health", 2, 5, 2)
        Daily_Travel_Time = st.slider("Daily Travel Time (Min)", 2, 65, 10)
        Likes_Subject = st.radio("Likes Subject", ["Yes", "No"])
        Group_Study = st.radio("Group Study", ["Yes", "No"])

        Likes_Subject_num = 1 if Likes_Subject == "Yes" else 0
        Group_Study_num = 1 if Group_Study == "Yes" else 0

   
    pred_placeholder = st.empty()

    if st.button("Predict", key="predict_button"):
        input_data = np.array([[Hours_Studied,
                                Previous_Grade,
                                Attendance,
                                Play_hours,
                                Sleep_hours,
                                Daily_Travel_Time,
                                Likes_Subject_num,
                                Health,
                                Group_Study_num]])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            pred_placeholder.markdown(
                f"<div class='custom-box pass-box'>👨🏻‍🎓 The student is <b>likely to PASS</b> 🎉</div>",
                unsafe_allow_html=True
            )
        else:
            pred_placeholder.markdown(
                f"<div class='custom-box fail-box'>👨🏻The student is <b>likely to FAIL</b> </div>",
                unsafe_allow_html=True
            )

        time.sleep(3.5)
        pred_placeholder.empty()
with tab_2:
    st.markdown("""
    ###  About the App
    This app is built with **Random Forest model** trained to predict whether a student is likely to pass or fail :
    
  
    #### 🔍 How It Works
    The model was trained using  student data and evaluates multiple academic and personal lifestyle indicators such as:
                   
    - 📚 **Hours Studied** daily
    - 🏆 **Previous Academic Grades**
    - 🧾 **Class Attendance Percentage**
    - 😴 **Sleep Patterns**
    - 🕹️ **Play Time vs Study Balance**
    - 🚍 **Daily Travel Time**
    - 💡 **Interest in Subject**
    - 🧑‍🤝‍🧑 **Group Study Participation**
    - ❤️ **Overall Health Score**

    ---
    #### 🧠 Model Architecture
      🧪 **Data Preprocessing**
       - Feature selection based on importance
       - Label Encoding and One-Hot Encoding for categorical variables
                
     🔬**Random Forest Classifier**
        - Ensemble based algorithm with multiple decision trees
        - Provides better generalization (low bias & low variance)
    
     🧮 **Performance evaluated**
                - cross-validation, confusion matrix, and F1-score
    """)
