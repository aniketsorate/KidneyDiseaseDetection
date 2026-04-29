import streamlit as st
from PIL import Image

from genai.workflow import KidneyReportWorkflow
from ml.predictor import predict_image


st.set_page_config(page_title="Kidney Disease Detection", page_icon="K", layout="wide")


@st.cache_resource
def get_workflow():
    return KidneyReportWorkflow()


with st.sidebar:
    st.header("About This Tool")
    st.info(
        "This AI-powered tool helps identify potential kidney abnormalities such as "
        "**Cysts, Tumors, and Stones**. It now combines a CNN image model with a "
        "LangChain prompt-based GenAI explanation workflow."
    )
    st.warning(
        "**Disclaimer:** This tool provides preliminary educational results and "
        "should not be used as a definitive diagnosis. Consult a healthcare "
        "professional for further evaluation."
    )
    st.caption("Stack: TensorFlow, FastAPI, LangChain prompts, LangGraph workflow.")


st.title("Kidney Disease Detection")
st.markdown("### Upload an image to detect abnormalities and generate an AI report.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    prediction = predict_image(img)
    report = get_workflow().run(prediction)
    predicted_class = prediction["diagnosis"]

    st.markdown("## Diagnosis Result")

    if predicted_class.lower() == "normal":
        diagnosis_message = (
            f'<p style="font-size:32px; font-weight:bold; color:#008000;">'
            f"Diagnosed as: {predicted_class}</p>"
        )
    else:
        diagnosis_message = (
            f'<p style="font-size:32px; font-weight:bold; color:#FF0000;">'
            f"Detected: {predicted_class}</p>"
        )

    st.markdown(diagnosis_message, unsafe_allow_html=True)
    st.metric("Confidence", f"{prediction['confidence']:.2%}", report["confidence_level"].title())

    left, right = st.columns([1, 1])
    with left:
        st.image(img, caption="Uploaded Image", use_container_width=True)
    with right:
        st.subheader("Class Probabilities")
        st.bar_chart(prediction["probabilities"])

    st.markdown("## GenAI Report")
    st.write(report["ai_explanation"])

    with st.expander("Prompt sent to the GenAI layer"):
        st.markdown("**System prompt**")
        st.code(report["system_prompt"])
        st.markdown("**User prompt**")
        st.code(report["user_prompt"])

    st.info(report["safety_note"])
    st.caption(f"Generation mode: {report['generation_mode']}")
