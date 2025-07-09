# app.py

import streamlit as st
import numpy as np
from PIL import Image

from canvas import get_digit_from_canvas
from predict_single import predict_single_digit

def preprocess_canvas(img):
    if img is None:
        return None

    img = Image.fromarray((255 - img[:, :, 0]).astype('uint8'))  # Red channel, invert
    img = img.resize((28, 28))
    img = np.array(img).astype('float32') / 255.0
    return img.reshape(1, 784)

st.set_page_config(page_title="MNIST Digit Identifier", layout="centered")
st.title("🧠 Handwritten Digit Recognizer")

image_data = get_digit_from_canvas()

if st.button("🔍 Predict Digit"):
    processed = preprocess_canvas(image_data)
    if processed is not None:
        label, conf, all_preds = predict_single_digit(
            processed,
            model_path="model/model_params.pkl",
            hidden_layers=[256, 64],
            output_size=10
        )
        st.success(f"Predicted: {label} with {conf*100:.2f}% confidence")
        st.bar_chart(all_preds[0])
    else:
        st.warning("Please draw a digit first.")
