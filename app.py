import streamlit as st
from PIL import Image
from pipeline.predict import predict_class, class_names

# Set Page Configuration
st.set_page_config(page_title="Kidney Disease Detection", page_icon="🩺", layout="wide")

# Sidebar for Additional Information
with st.sidebar:
    st.header("ℹ️ About This Tool")
    st.info(
        "This AI-powered tool helps identify potential kidney abnormalities such as **Cysts, Tumors, and Stones**. "
        "However, it is **not a substitute for professional medical advice**. Please consult a doctor for accurate diagnosis."
    )
    
    # Disclaimer
    st.warning(
        "🔍 **Disclaimer:** This tool provides preliminary results and should not be used as a definitive diagnosis. "
        "Consult a healthcare professional for further evaluation."
    )

# Title and Description
st.title("🩺 Kidney Disease Detection")
st.markdown("### Upload an image to detect any abnormalities in the kidney.")

# Upload Image
uploaded_file = st.file_uploader("📸 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    img = Image.open(uploaded_file)

    # Get Prediction
    predicted_class, _, _ = predict_class(img)

    # Diagnosis Message with Larger Font Size
    if predicted_class.lower() == "normal":
        diagnosis_message = (
            f'<p style="font-size:32px; font-weight:bold; color:#008000;">✅ Diagnosed as: {predicted_class}</p>'
        )
    else:
        diagnosis_message = (
            f'<p style="font-size:32px; font-weight:bold; color:#FF0000;">⚠️ Diagnosed as: {predicted_class}</p>'
        )

    # Display Diagnosis Message at the TOP
    st.markdown("## 🏥 **Diagnosis Result**")
    st.markdown(diagnosis_message, unsafe_allow_html=True)

    # Extra Warning if Cyst is Detected
    if predicted_class.lower() == "cyst":
        st.warning("⚠️ **Cysts can sometimes indicate an underlying tumor.** Further medical evaluation is recommended.")

    # Display Uploaded Image
    st.image(img, caption="🖼️ Uploaded Image", use_container_width=True, width=300)
