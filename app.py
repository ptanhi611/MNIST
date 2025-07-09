import streamlit as st
import numpy as np
from PIL import Image

from canvas import get_digit_from_canvas
from predict_single import predict_single_digit

# ---------- Preprocessing ----------
def preprocess_canvas(img):
    if img is None:
        return None

    img = Image.fromarray((255 - img[:, :, 0]).astype('uint8'))  # Grayscale
    img = img.resize((28, 28))
    img = np.array(img).astype('float32') / 255.0
    return img.reshape(1, 784)


# ---------- Page Config ----------
st.set_page_config(page_title="MNIST Digit Recognizer", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4A90E2;'>✍️ Handwritten Digit Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Draw a digit (0–9) using mouse, touchpad, or touchscreen</p>", unsafe_allow_html=True)

# ---------- Layout ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Drawing Canvas")
    canvas_result = get_digit_from_canvas()

    # Add "Clear Canvas" workaround using rerun button
    if st.button("🧹 Clear Canvas"):
        st.rerun()

with col2:
    st.markdown("### Prediction Result")

    if st.button("🔍 Predict Digit"):
        processed = preprocess_canvas(canvas_result)
        if processed is not None:
            label, conf, all_preds = predict_single_digit(
                processed,
                model_path="model_params_stored_mnist.pk1",
                hidden_layers=[128, 64],  # Change if needed
                output_size=10
            )

            st.success(f"✅ Predicted Digit: **{label}**")
            st.markdown(f"<p style='font-size:16px'>Confidence: <strong>{conf*100:.2f}%</strong></p>", unsafe_allow_html=True)
            st.bar_chart(all_preds[0])
        else:
            st.warning("Please draw a digit on the canvas first.")

