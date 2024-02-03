import streamlit as st
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
from PIL import Image

# Load the pre-trained ResNet model and feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")

# Function to preprocess the image and make predictions
def classify_image(image):
    inputs = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_label]
    return predicted_class

# Streamlit app
st.title("Image Classification with ResNet-18")

# File uploader for user to upload their image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Display the uploaded image
if uploaded_file is not None:
    # Preprocess the image
    image = Image.open(uploaded_file).convert("RGB")

    # Make predictions
    predicted_class = classify_image(image)

    # Display the image and predictions
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Label: {predicted_class}")