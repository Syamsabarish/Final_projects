import streamlit as st
import pandas as pd
import joblib

# Load model once
@st.cache_resource
def load_model():
    bundle = joblib.load(r"C:\Users\SYAMNARAYANAN\OneDrive\Desktop\visual code\Guvi_Project_3\model_clustering.joblib")
    return bundle["kmeans"], bundle["scaler"]

kmeans, scaler = load_model()

# Add background CSS globally
st.markdown("""<style>
 .stApp {
    background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.8)), 
                      url("https://static.vecteezy.com/system/resources/previews/001/311/554/original/education-and-back-to-school-concept-free-vector.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.main > div {
    background-color: rgba(0, 0, 0, 0.6);
    padding: 2rem;
    border-radius: 10px;
}
h2, p, li, div {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

label_map = {
    0: "Excellent Learner",
    1: "Good Learner",
    2: "Average Learner"
}


st.sidebar.title("Direction")
page = st.sidebar.selectbox("Choose a page", ["1️⃣ Intro", "2️⃣ Prediction", "3️⃣ Info"])

if page == "1️⃣ Intro":
    st.title("🎓 Student Learning Style Clustering App")

    st.markdown("""
    Welcome to the **Student Learning Style Predictor** built using K-Means clustering and Streamlit.

    This app groups students into:
    - **Excellent Learner**
    - **Good Learner**
    - **Average Learner**

     Based on features like:
    - Reading speed
    - Interaction time
    - Quiz accuracy
    - Video watched %
    - Topics mastered
    """)

elif page == "2️⃣ Prediction":
    st.title("🎯 Predict Student Learning Style")

    reading_speed = st.slider("📖 Reading Speed", 50.0, 400.0, 190.0)
    interaction_time = st.slider("⏱ Interaction Time", 50.0, 800.0, 350.0)
    quiz_accuracy = st.slider("🧪 Quiz Accuracy", 0.0, 1.0, 0.7)
    video_watched_percent = st.slider("🎥 Video Watched %", 0.0, 1.0, 0.9)
    topics_mastered = st.slider("📘 Topics Mastered", 0, 20, 7)

    user_input = pd.DataFrame({
        'reading_speed': [reading_speed],
        'interaction_time': [interaction_time],
        'quiz_accuracy': [quiz_accuracy],
        'video_watched_percent': [video_watched_percent],
        'topics_mastered': [topics_mastered]
    })

    if st.button("🔍 Predict"):
        scaled_input = scaler.transform(user_input)
        cluster = kmeans.predict(scaled_input)[0]
        learner_type = label_map.get(cluster, f"Group {cluster}")

        st.success(f"🎯 Predicted Learning Style: **{learner_type}**")
        st.write("🔢 Cluster:", cluster)
        st.write("🧾 Input Data:")
        st.dataframe(user_input)


elif page == "3️⃣ Info":
    
    st.markdown("## ℹ️ About This Project")

    st.markdown(""" 
    - 🛠 **Tech Stack**: Python, Streamlit, Scikit-Learn  
    - 📊 **Model**: K-Means Clustering for student learning styles  
    

    ### How the Model Works

    This app uses **K-Means Clustering**, an unsupervised machine learning technique, to group students based on their learning behaviors.

    - It analyzes features like reading speed, interaction time, quiz accuracy, video watched percentage, and topics mastered.
    - The algorithm automatically finds 3 groups:  
      **Excellent Learners**, **Good Learners**, and **Average Learners**.
    - When you input your data, the model predicts which learning group.
    """)

