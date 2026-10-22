
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image, UnidentifiedImageError

from extract import hide_message, extract_message
from predict import SteganographyPredictor


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "lsb_random_forest.joblib"

st.set_page_config(
    page_title="AI Steganography Detection",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #eaf3ff, #f4f0ff);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        color: #12345a;
    }

    .muted {
        color: #53657a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🔎 AI-Based Steganography Detection</h1>
        <p class="muted">
            Analyze images using LSB statistical features and a
            Random Forest classifier, recover supported hidden
            messages, and encode your own secret messages.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This project uses an LSB feature-based Random Forest classifier. "
    "Message extraction checks this project's supported LSB payload "
    "format. A Clean prediction does not guarantee that an image "
    "contains no hidden information."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_upload(uploaded_file):
    """Save an uploaded file temporarily and return its path."""

    suffix = Path(uploaded_file.name).suffix.lower() or ".png"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temporary_file:
        temporary_file.write(uploaded_file.getvalue())
        return Path(temporary_file.name)


@st.cache_resource
def load_predictor(model_path_string):
    """Load and cache the Random Forest predictor."""

    predictor = SteganographyPredictor(
        model_path=model_path_string
    )

    if not predictor.load_model():
        raise RuntimeError(
            "The Random Forest model could not be loaded."
        )

    return predictor


def get_predictor():
    """Return the predictor or show a useful error."""

    if not MODEL_PATH.exists():
        st.error(
            f"Random Forest model not found: {MODEL_PATH}"
        )
        st.info(
            "Run python test_lsb_classifier.py to train and "
            "save the model first."
        )
        return None

    try:
        return load_predictor(str(MODEL_PATH))
    except Exception as error:
        st.error(f"Could not load the Random Forest model: {error}")
        return None


# ============================================================
# TABS
# ============================================================

tab_detect, tab_encode, tab_about = st.tabs(
    [
        "🔍 Analyze image",
        "✉️ Hide a message",
        "ℹ️ Project details",
    ]
)


# ============================================================
# TAB 1: ANALYZE IMAGE
# ============================================================

with tab_detect:
    st.subheader("Analyze an image")

    uploaded = st.file_uploader(
        "Upload an image to analyze",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        key="detect_upload",
    )

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")

            left, right = st.columns([1.2, 1])

            with left:
                st.image(
                    image,
                    caption=uploaded.name,
                    use_container_width=True,
                )

            with right:
                st.write("**Image details**")
                st.write(f"File name: {uploaded.name}")
                st.write(
                    f"Dimensions: {image.width} × {image.height}"
                )
                st.write(
                    f"File type: {uploaded.type or 'Unknown'}"
                )

                if st.button(
                    "Analyze image",
                    type="primary",
                    use_container_width=True,
                    key="analyze_button",
                ):
                    temporary_path = None

                    try:
                        temporary_path = save_upload(uploaded)

                        # ------------------------------------
                        # STEP 1: RANDOM FOREST CLASSIFICATION
                        # ------------------------------------

                        st.markdown("---")
                        st.subheader("Random Forest classification")

                        predictor = get_predictor()

                        if predictor is not None:
                            try:
                                result = predictor.predict_single_image(
                                    str(temporary_path)
                                )

                                if (
                                    isinstance(result, (tuple, list))
                                    and len(result) >= 2
                                ):
                                    label, score = result[0], result[1]
                                else:
                                    label, score = result, None

                                label_text = str(label).strip()
                                normalized_label = label_text.lower()

                                if "stego" in normalized_label:
                                    st.warning(
                                        "Prediction: Stego"
                                    )
                                elif "clean" in normalized_label:
                                    st.success(
                                        "Prediction: Clean"
                                    )
                                else:
                                    st.info(
                                        f"Prediction: {label_text}"
                                    )

                                st.metric(
                                    "Prediction",
                                    label_text,
                                )

                                if score is not None:
                                    try:
                                        confidence = float(score)

                                        # Supports either probability
                                        # values such as 0.83 or
                                        # percentages such as 83.
                                        if confidence > 1:
                                            confidence /= 100.0

                                        confidence = max(
                                            0.0,
                                            min(1.0, confidence),
                                        )

                                        st.metric(
                                            "Model confidence",
                                            f"{confidence:.1%}",
                                        )
                                    except (TypeError, ValueError):
                                        st.caption(
                                            f"Model score: {score}"
                                        )

                                st.caption(
                                    "The model score is not a guarantee "
                                    "that the prediction is correct."
                                )

                            except Exception as error:
                                st.error(
                                    "Classification failed: "
                                    f"{error}"
                                )

                        # ------------------------------------
                        # STEP 2: LSB MESSAGE EXTRACTION
                        # ------------------------------------

                        st.markdown("---")
                        st.subheader("LSB payload check")

                        try:
                            message = extract_message(
                                str(temporary_path)
                            )

                            if message is not None:
                                st.success(
                                    "Message recovered"
                                )

                                st.metric(
                                    "LSB payload check",
                                    "Message recovered",
                                )

                                st.text_area(
                                    "Recovered message",
                                    value=message,
                                    height=150,
                                    key="recovered_message",
                                )

                            else:
                                st.warning(
                                    "No valid message in this project's "
                                    "LSB format was recovered. This does "
                                    "not prove the image is clean."
                                )

                                st.metric(
                                    "LSB payload check",
                                    "No verified message",
                                )

                        except Exception as error:
                            st.warning(
                                "Message extraction could not be "
                                f"completed: {error}"
                            )

                    except (OSError, ValueError) as error:
                        st.error(f"Could not analyze the image: {error}")

                    finally:
                        if temporary_path is not None:
                            temporary_path.unlink(missing_ok=True)

        except (UnidentifiedImageError, OSError) as error:
            st.error(f"Could not read this image: {error}")


# ============================================================
# TAB 2: HIDE A MESSAGE
# ============================================================

with tab_encode:
    st.subheader("Hide a secret message")

    st.write(
        "Embed text in an image using this project's LSB encoding "
        "format. Download the resulting image as PNG to preserve "
        "the embedded bits."
    )

    cover = st.file_uploader(
        "Choose a cover image",
        type=["png", "bmp", "jpg", "jpeg"],
        key="encode_upload",
    )

    secret = st.text_area(
        "Message to hide",
        placeholder="Type your message here...",
        max_chars=10000,
        key="secret_message",
    )

    if cover is not None:
        try:
            cover_image = Image.open(cover).convert("RGB")

            st.image(
                cover_image,
                caption="Cover image preview",
                width=360,
            )

            capacity = max(
                0,
                (cover_image.width * cover_image.height * 3 - 64) // 8,
            )

            st.caption(
                f"Approximate maximum message size: "
                f"{capacity:,} bytes, including header overhead."
            )

            if st.button(
                "Encode and prepare download",
                type="primary",
                key="encode_button",
            ):
                if not secret:
                    st.error("Enter a message before encoding.")

                elif len(secret.encode("utf-8")) > capacity:
                    st.error(
                        "The message may be too large for this image. "
                        "Use a larger image or a shorter message."
                    )

                else:
                    source_path = None
                    output_path = None

                    try:
                        source_path = save_upload(cover)

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".png",
                        ) as temporary_output:
                            output_path = Path(temporary_output.name)

                        success = hide_message(
                            str(source_path),
                            secret,
                            str(output_path),
                        )

                        if success and output_path.exists():
                            verification = extract_message(
                                str(output_path)
                            )

                            if verification == secret:
                                st.success(
                                    "Message embedded and verified "
                                    "successfully!"
                                )
                            else:
                                st.warning(
                                    "An output image was created, but "
                                    "the extracted message did not "
                                    "match the original text."
                                )

                            st.image(
                                str(output_path),
                                caption="Generated stego image",
                                width=360,
                            )

                            st.download_button(
                                "⬇️ Download stego PNG",
                                data=output_path.read_bytes(),
                                file_name="stego_image.png",
                                mime="image/png",
                                key="download_stego",
                            )

                        else:
                            st.error(
                                "Encoding failed. Try a larger image "
                                "or a shorter message."
                            )

                    except Exception as error:
                        st.error(f"Could not encode the message: {error}")

                    finally:
                        if source_path is not None:
                            source_path.unlink(missing_ok=True)

                        if output_path is not None:
                            output_path.unlink(missing_ok=True)

        except (UnidentifiedImageError, OSError) as error:
            st.error(f"Could not read the cover image: {error}")


# ============================================================
# TAB 3: PROJECT DETAILS
# ============================================================

with tab_about:
    st.subheader("How it works")

    st.markdown(
        """
        ### 1. LSB encoding and extraction
        The system can embed and recover text using this project's
        supported Least Significant Bit (LSB) payload format.

        ### 2. Random Forest detection
        The classifier uses three features:
        - LSB ones ratio
        - Horizontal LSB changes
        - Vertical LSB changes

        ### 3. Prediction results
        The model predicts Clean or Stego and may provide a
        confidence score. Some images can be misclassified.

        ### 4. Dataset and evaluation
        The current experimental dataset contains synthetic images.
        Performance on this dataset does not establish accuracy on
        real-world steganographic images.

        ### Current model
        `model/lsb_random_forest.joblib`
        """
    )

    st.markdown("---")
    st.subheader("Model status")

    if MODEL_PATH.exists():
        st.success("Random Forest model file found.")

        with st.expander("View model information"):
            try:
                predictor = get_predictor()

                if predictor is not None:
                    if hasattr(predictor, "get_model_info"):
                        st.json(predictor.get_model_info())
                    else:
                        st.write(
                            "Model loaded successfully. "
                            "Detailed model information is not "
                            "available through this predictor method."
                        )

            except Exception as error:
                st.warning(
                    f"Could not retrieve model information: {error}"
                )
    else:
        st.error("Random Forest model file not found.")
        st.code(
            "python test_lsb_classifier.py",
            language="powershell",
        )

    st.markdown("---")
    st.subheader("Run the application")

    st.code(
        "streamlit run app.py",
        language="powershell",
    )
