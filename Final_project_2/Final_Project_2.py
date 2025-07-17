import streamlit as st
import numpy as np
import joblib

model = joblib.load(r'C:\Users\SYAMNARAYANAN\OneDrive\Desktop\visual code\Guvi_Project_3\PROJECT_2_SCORE_PRED_3.pkl')



st.markdown("""
    <style>
        .stApp {
            background-image: url("https://static.vecteezy.com/system/resources/previews/006/691/884/original/blue-question-mark-background-with-text-space-quiz-symbol-vector.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            color: black;
        }
        .stTabs{
            color: black;
            font-weight: bold;

    }      
    </style>
    """,
    unsafe_allow_html=True
)

parent_map = {"High School": 0, "Bachelor": 1, "Master": 2}
internet_map = {"Poor": 0, "Average": 1, "Good": 2}


tab1, tab2, tab3 = st.tabs(["Home", "Prediction", "About"])

# page1
with tab1:
    st.title("Student Score Predictor")
    st.markdown("""
This tool is designed to help predict a student's **academic performance** based on various lifestyle.

###  What does this app do?

By analyzing patterns in key inputs such as:
- Study time
- Health
- Attendance
- Parental education level
- Internet access
- And more.

...The app provides an estimated **final grade** for a student. """)



# page2
with tab2:
    st.header("Enter Student Information")
    
    study_hours = st.slider("Study Hours Per Day", 0, 12, 2)
    mental_health_rating = st.number_input("Mental Health Rating (0–8)", 0, 8,3)
    social_hours = st.slider("Social Media Hours", 0, 12, 2)
    netflix_hours = st.slider("Netflix Hours", 0, 10, 1)
    sleep_hours = st.slider("Sleep Hours", 0, 12, 6)
    exercise_frequency = st.slider("Exercise Frequency (per week)", 0, 6, 2)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    age = st.number_input("Age", 15, 25, 20)

    parental_education_level = st.selectbox("Parental Education Level", list(parent_map.keys()))
    Diet_quality = st.selectbox("Diet Quality", list(internet_map.keys()))

    st.sidebar.header("See Your Predicted Score")
    if st.sidebar.button("MARKS"):
        parent_enc = parent_map[parental_education_level]
        internet_enc = internet_map[Diet_quality]

        X_new = np.array([[study_hours, mental_health_rating, social_hours, netflix_hours,
                           sleep_hours, exercise_frequency, attendance, age,
                           parent_enc, internet_enc]])

        pred = model.predict(X_new)[0]
        st.sidebar.markdown("---")
        if pred >= 80:
            st.sidebar.success(f"**Predicted Score:** {pred:.2f}\nCongratulations!")
        elif pred >=40:
            st.sidebar.info(f"**Predicted Score:** {pred:.2f}\n Good! Do better")
        else:
            st.sidebar.error(f"**Predicted Score:** {pred:.2f}\nConsider reviewing study habits.")
# page3
with tab3:
    st.title("About This App")
    st.markdown("""
This project uses a **GradientBoostingRegressor** to predict students' performance based on:

- Study habits
- Sleep & exercise
- Mental health & social media
- Parental education
- Internet quality and More...
- Uses Joblib to load model(.Pkl) """)
