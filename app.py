import streamlit as st
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

st.title('Glaucoma Risk Prediction')

# Load the saved model
model_path = 'EffNet-model.h5'
model = load_model(model_path)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    img_size = (224, 224)
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)

    # Get the predicted class
    predicted_class_index = np.argmax(prediction)

    # Map class index to class label
    class_labels = {0: 'HEALTHY', 1: 'MILD ', 2: 'MODERATE', 3: 'PROLIFERATE', 4: 'SEVERE'}
    predicted_class = class_labels[predicted_class_index]


    # Display prediction result
    st.image(uploaded_file)
    st.write(f"Predicted Glaucoma Risk: {predicted_class}")
