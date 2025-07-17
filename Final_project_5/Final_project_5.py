import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Disable oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

@st.cache_resource
def load_all():
    model = load_model(r'C:\Users\SYAMNARAYANAN\OneDrive\Desktop\visual code\Guvi_Project_3\subject_classifier.h5')
    tokenizer = joblib.load(r'C:\Users\SYAMNARAYANAN\OneDrive\Desktop\visual code\Guvi_Project_3\tokenizer.pkl')
    label_encoder = joblib.load(r'C:\Users\SYAMNARAYANAN\OneDrive\Desktop\visual code\Guvi_Project_3\label_encoder.pkl')
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_all()


def set_bg():
    st.markdown(
        '''
        <style>
        .stApp {
            background-image: linear-gradient(rgba(0,0,0,0.2), rgba(0,0,0,0.2)),
            url("https://images.rawpixel.com/image_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIzLTAzL3JtNjA0YmF0Y2gyLWZyYW1lLWJtLTAyLTAxLWEuanBn.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;
            color: black;
        }
         div.stButton > button {
        background-color: #4CAF50; 
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
        </style>
        ''', unsafe_allow_html=True
    )

set_bg()

if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    with st.container():
        st.title("📚 The Subject Classifier App")
        st.markdown("""
        Welcome to the **Student Subject Classifier**!

        ### What is this app about?
        This interactive web app allows you to classify student-written paragraphs into their respective subjects using state-of-the-art deep learning models.

        """)

elif st.session_state.page == "Predict":
    with st.container():
        st.title("📘 Student Paragraph Subject Classifier")
        st.write("Enter a paragraph below and the model will predict its academic subject.")

        user_input = st.text_area("Your text here:", height=150)

        if st.button("Predict Subject"):
            if user_input.strip() == "":
                st.warning("Please enter some text to classify.")
            else:
                seq = tokenizer.texts_to_sequences([user_input])
                padded = pad_sequences(seq, maxlen=200)
                preds = model.predict(padded)
                predicted_index = preds.argmax()
                predicted_label = label_encoder.inverse_transform([predicted_index])[0]
                st.success(f"**Predicted Subject:** {predicted_label}")

elif st.session_state.page == "Info":
    with st.container():
        st.title("📖 About This App")
        st.markdown("""
        This app uses a **deep learning model (LSTM)** to classify student paragraphs based on subject matter.

        ### How does it work?
        - The model was trained on labeled student text using TensorFlow/Keras.
        - Text is tokenized, padded, and passed through an LSTM to predict the most relevant subject.

        ### Why use it?
        - Automate large-scale student essay classification.
        - Explore NLP techniques in an intuitive interface.
        - Built with **Streamlit**, **TensorFlow**, and **Python**.
        """)

with st.container():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button( "Home"):
            st.session_state.page = "Home"
            st.rerun()

    with col2:
        if st.button("Predict"):
            st.session_state.page = "Predict"
            st.rerun()

    with col3:
        if st.button(" Info"):
            st.session_state.page = "Info"
            st.rerun()
