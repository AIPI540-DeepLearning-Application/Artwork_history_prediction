import streamlit as st
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
from datasets import load_dataset

# Load example image from the dataset
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

# Load feature extractor and model
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")

# Preprocess the image and make predictions
inputs = feature_extractor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted label
predicted_label = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_label]

# Display the result using Streamlit
st.image(image, caption="Input Image", use_column_width=True)
st.write(f"Predicted Label: {predicted_class}")
