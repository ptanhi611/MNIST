
from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
from PIL import Image, ImageOps

def get_digit_from_canvas():
    st.markdown("### ✏️ Draw a digit below")

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=25,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Convert to grayscale
        img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype(np.uint8)).convert("L")

        # Invert (black on white → white on black)
        img = ImageOps.invert(img)

        # Crop the digit using bounding box
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        # Add padding to make it square
        max_side = max(img.size)
        padded_img = Image.new("L", (max_side, max_side), 0)
        offset = ((max_side - img.size[0]) // 2, (max_side - img.size[1]) // 2)
        padded_img.paste(img, offset)

        # Resize to MNIST size
        img = padded_img.resize((28, 28), Image.LANCZOS)

        # Normalize
        img = np.array(img).astype(np.float32)
        img = img / 255.0
        img = img.reshape(1, -1)

        # Standardize (mean 0, std 1)
        mean = img.mean()
        std = img.std() if img.std() > 0 else 1
        img = (img - mean) / std

        return img
    return None
