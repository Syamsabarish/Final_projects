import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import streamlit as st
from transformers import pipeline

def set_background(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def intro_page():
    st.title("Welcome to the TED Talk Tagger App")
    st.write("This app allows you to enter a TED Talk description and predicts relevant tags using AI.")
    st.write("Navigate to the **Prediction** page to try it out.")

def prediction_page():
    st.title("TED Talk Description Tagger")
    st.markdown("Enter a TED Talk description, and get predicted tags!")

    user_input = st.text_area("Enter TED Talk description:", height=200)

    if st.button("Get Tags"):
        if not user_input.strip():
            st.warning("Please enter some description text.")
            return

        classifier = load_model()

        candidate_tags = [
            "technology", "science", "education", "health", "business",
            "art", "design", "psychology", "environment", "innovation",
            "culture", "motivation", "history", "politics", "music",
            "philosophy", "AI", "robotics", "space", "economics"
        ]

        with st.spinner("Classifying..."):
            result = classifier(user_input, candidate_tags, multi_label=True)

        threshold = 0.5
        tags = [tag for tag, score in zip(result['labels'], result['scores']) if score > threshold]

        if tags:
            st.success("Predicted Tags:")
            st.write(", ".join(tags))
        else:
            st.info("No tags matched with high confidence.")

def info_page():
    st.title("About This App")
    st.write("""
    This app uses Hugging Face's zero-shot classification model (`facebook/bart-large-mnli`) to predict tags for TED Talk descriptions.
    
    You enter a description, and the model predicts which tags best fit the content from a predefined list.
    
    This is a demo project showcasing natural language processing and zero-shot classification.
    """)


def main():
    image_url = "https://wallpaperaccess.com/full/1880555.jpg"
    set_background(image_url)

    st.sidebar.title("MAP")
    page = st.sidebar.radio("Go to", ["Intro", "Prediction", "Info"])

    if page == "Intro":
        intro_page()
    elif page == "Prediction":
        prediction_page()
    elif page == "Info":
        info_page()

if __name__ == "__main__":
    main()
