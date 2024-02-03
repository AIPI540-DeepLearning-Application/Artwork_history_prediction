import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import json

# load labels
with open('imagenet_labels.json') as f:
    labels = json.load(f)

def main():
    st.title("PyTorch Image Classification")
    # upload images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # show images
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # classify
        model = load_model()
        prediction = classify_image(image, model)
        st.write(f"Prediction: {prediction}")

def load_model():
    # load
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

def classify_image(image, model):
    # pre processing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0)
    # predict
    with torch.no_grad():
        output = model(image)
        _, predicted_idx = torch.max(output, 1)
        predicted_label = labels[str(predicted_idx.item())]
    return predicted_label

if __name__ == '__main__':
    main()