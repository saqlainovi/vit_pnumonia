import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ViTForImageClassification
from torchvision import transforms
import os

# Page config
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# Title and description
st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to detect the presence of pneumonia.")

@st.cache_resource
def load_model():
    # Initialize model
    device = torch.device('cpu')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)

    # Load trained weights if available
    model_path = './model/best_pneumonia_model.pth'
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded trained model successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using base model instead.")

    model.eval()
    return model, device

# Load model
model, device = load_model()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_pneumonia(image):
    if image is None:
        return None
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor).logits
        probabilities = F.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item() * 100
    
    return {
        "Normal": float(probabilities[0][0]),
        "Pneumonia": float(probabilities[0][1])
    }

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
    
    # Add a prediction button
    if st.button('Analyze Image'):
        with st.spinner('Analyzing...'):
            # Get prediction
            results = predict_pneumonia(image)
            
            # Display results
            st.subheader("Results:")
            col1, col2 = st.columns(2)
            
            # Normal probability
            with col1:
                normal_prob = results["Normal"] * 100
                st.metric("Normal", f"{normal_prob:.1f}%")
                
            # Pneumonia probability
            with col2:
                pneumonia_prob = results["Pneumonia"] * 100
                st.metric("Pneumonia", f"{pneumonia_prob:.1f}%")
            
            # Final prediction
            prediction = "Pneumonia" if results["Pneumonia"] > results["Normal"] else "Normal"
            confidence = max(results["Pneumonia"], results["Normal"]) * 100
            
            st.subheader("Final Assessment:")
            st.write(f"The X-ray image appears to be: **{prediction}**")
            st.write(f"Confidence: **{confidence:.1f}%**")

# Add disclaimer
st.markdown("""
---
**Important Note:** This is an AI-assisted analysis tool and should not be used as the sole basis for diagnosis. 
Please consult with a qualified healthcare professional for proper diagnosis and treatment.
""") 