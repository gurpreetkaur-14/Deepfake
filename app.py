
import streamlit as st
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #0B1120;
    color: white;
}

h1, h2, h3 {
    color: #FFFFFF;
}

.subtitle {
    color: #A5B4CC;
    font-size: 17px;
}

.upload-box {
    background-color: #172238;
    padding: 25px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔍 DeepFake Detector")

st.markdown(
    '<p class="subtitle">AI-Powered Image Authenticity Analysis</p>',
    unsafe_allow_html=True
)

st.divider()

# Introduction
st.subheader("Welcome to DeepFake Detector")

st.write(
    "Upload a facial image to analyse its authenticity "
    "using our proposed deep learning system."
)

# Project information
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Detection Classes", "2")

with col2:
    st.metric("Proposed Model", "MobileNetV2")

with col3:
    st.metric("Analysis Type", "Image")

st.divider()

# Main interface
left, right = st.columns([1, 1])

with left:
    st.subheader("Upload Your Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.success("Image uploaded successfully!")

        analyse = st.button(
            "Analyse Image",
            use_container_width=True
        )

        if analyse:
            st.info(
                "Frontend demonstration: "
                "The deep learning model is not integrated yet. "
                "Model training and integration are pending."
            )

with right:
    st.subheader("Detection Results")

    st.info(
        "Your prediction will appear here after "
        "the trained model is integrated."
    )

    st.markdown("### Model Information")

    st.write("Architecture: MobileNetV2")

    st.write("Task: Binary Image Classification")

    st.write("Output Classes: Real / Fake")

    st.write("Input Size: 224 × 224")

st.divider()

# Footer
st.caption(
    "Minor Project | B.Tech CSE (Artificial Intelligence) | "
    "Academic Session 2026–27"
)

st.caption(
    "Note: AI predictions are estimates and should not "
    "be treated as definitive proof of authenticity."
)