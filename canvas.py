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

        # Invert to get white digit on black background
        img = ImageOps.invert(img)

        # Crop to content
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        # Pad to square
        max_side = max(img.size)
        square_img = Image.new("L", (max_side, max_side), 0)
        square_img.paste(img, ((max_side - img.size[0]) // 2, (max_side - img.size[1]) // 2))

        # Resize to MNIST dimensions
        img = square_img.resize((28, 28), Image.LANCZOS)

        # Convert to numpy and binarize faint strokes
        img = np.array(img).astype(np.uint8)
        img = np.where(img > 30, 255, 0).astype(np.uint8)

        # Normalize using fixed MNIST stats
        img = img.astype(np.float32) / 255.0
        img = (img - 0.1307) / 0.3081

        return img.reshape(1, -1)

    return None
