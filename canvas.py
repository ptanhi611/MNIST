from streamlit_drawable_canvas import st_canvas
import streamlit as st

def get_digit_from_canvas():
    st.markdown("### ✏️ Draw a digit below")

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    return canvas_result.image_data if canvas_result.image_data is not None else None
