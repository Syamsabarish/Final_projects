import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model

@st.cache_resource

def load_trained_model():
    return tf.keras.models.load_model(r'Guvi_Project_3/mnist_model_6.keras')


import streamlit as st

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://cdn.pixabay.com/photo/2020/10/21/01/56/digital-5671888_960_720.png");
        background-size: cover;    
        background-repeat: no-repeat;
        background-position: center; 
        background-attachment: fixed; 
        color: #f7f9ff;
    }
    .stTabs {
        display: flex;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    </style>
    """,True
)

model = load_trained_model()

tab1, tab2, tab3 = st.tabs(["Home", "Predict Digit", "About"])


with tab1:
    st.title("Digit Recognizer App")
    st.markdown("""
        - App uses a **Convolutional Neural Network (CNN)** to recognize handwritten digits.

### Instructions:
- Go to the **Predict Digit** tab to upload an image and get a prediction.
- Visit the **About** tab to learn more about the model.

> Make sure your image is **28x28 pixels** and grayscale!
    """)


with tab2:
    st.title("Predict a Digit")

    uploaded_file = st.file_uploader("Upload a 28x28 grayscale image of a digit", type=["png", "jpg", "jpeg"])

    if uploaded_file:
       
        image = Image.open(uploaded_file).convert("L") 
        image = ImageOps.invert(image)                  
        image = image.resize((28, 28))                  
        st.image(image, caption=" Uploaded Image", width=150)

        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1).astype("float32")

        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
    

        st.success(f"Predicted Digit: {digit}")
       

with tab3:
    st.title("About")
    st.markdown("""
This app was built using:

- **Streamlit** for the UI
- **TensorFlow/Keras** for deep learning
- **PIL (Pillow)** and **NumPy** for image processing

### Model Details:
- Dataset: MNIST
- Type: CNN
- Input shape: (28x28x1 grayscale)
- Output: Probabilities for digits 0–9
    """)
