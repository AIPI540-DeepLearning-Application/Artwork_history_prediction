import streamlit as st
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
import json

def main():
    st.title("Artworks year prediction")
    
    # Load labels from JSON file
    labels = load_labels('./data/raw/art_labels.json')

    # Load model and feature extractor
    model, feature_extractor = load_model("AIPI540/art_predict")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Classify the image
        prediction_idx = classify_image(image, model, feature_extractor)
        prediction_label = labels[str(prediction_idx)]
        st.write(f"Prediction: {prediction_label}")

def load_labels(label_file):
    with open(label_file) as f:
        labels = json.load(f)
    return labels

def load_model(model_name):
    model = AutoModelForImageClassification.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor

def classify_image(image, model, feature_extractor):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx

if __name__ == '__main__':
    main()