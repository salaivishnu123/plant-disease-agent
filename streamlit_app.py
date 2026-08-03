import base64
import io
from pathlib import Path

from PIL import Image
import streamlit as st

from predict import ImageValidationError, PlantDiseasePipeline

st.set_page_config(page_title="Plant Disease Agent", layout="centered")

PAGE_DESCRIPTION = (
    "Upload a plant leaf image and get disease diagnosis, treatment guidance, and a Grad-CAM explanation. "
    "This Streamlit app reuses the existing model pipeline from `predict.py`."
)

st.title("Plant Disease Agent")
st.write(PAGE_DESCRIPTION)


@st.cache_resource
def load_pipeline() -> PlantDiseasePipeline:
    return PlantDiseasePipeline()


def decode_data_url(data_url: str) -> bytes:
    header, encoded = data_url.split(",", 1)
    return base64.b64decode(encoded)


def display_result(result: dict) -> None:
    st.subheader("Prediction")
    col1, col2 = st.columns(2)
    col1.metric("Plant", result["plant"], f"{result['plant_confidence']:.1%}")
    col2.metric("Disease", result["disease"], f"{result['disease_confidence']:.1%}")

    st.markdown(f"**Health status:** {result.get('health_status', 'Unknown')}")

    if result.get("symptoms"):
        st.markdown("**Symptoms:**")
        st.write(result["symptoms"])

    if result.get("causes"):
        st.markdown("**Causes:**")
        st.write(result["causes"])

    if any(result.get(key) for key in ["organic_treatment", "chemical_treatment", "fertilizer", "watering", "prevention", "future_care"]):
        st.markdown("### Treatment recommendations")
        if result.get("organic_treatment"):
            st.markdown(f"**Organic treatment:** {result['organic_treatment']}")
        if result.get("chemical_treatment"):
            st.markdown(f"**Chemical treatment:** {result['chemical_treatment']}")
        if result.get("fertilizer"):
            st.markdown(f"**Fertilizer:** {result['fertilizer']}")
        if result.get("watering"):
            st.markdown(f"**Watering:** {result['watering']}")
        if result.get("prevention"):
            st.markdown(f"**Prevention:** {result['prevention']}")
        if result.get("future_care"):
            st.markdown(f"**Future care:** {result['future_care']}")

    if result.get("explanation"):
        st.markdown("### Model explanation")
        st.write(result["explanation"])

    if result.get("gradcam_image"):
        st.markdown("### Grad-CAM")
        image_bytes = decode_data_url(result["gradcam_image"])
        st.image(Image.open(io.BytesIO(image_bytes)), caption="Grad-CAM overlay", use_column_width=True)


def main() -> None:
    pipeline = load_pipeline()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is None:
        st.info("Please upload a leaf image to get a prediction.")
        return

    file_bytes = uploaded_file.read()
    try:
        result = pipeline.predict(file_bytes)
    except ImageValidationError as exc:
        st.error(f"Invalid image: {exc}")
        return
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    with st.expander("Show input image"):
        st.image(Image.open(io.BytesIO(file_bytes)), caption="Uploaded image", use_column_width=True)

    display_result(result)


if __name__ == "__main__":
    main()
