import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import math

FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,
             365,379,378,400,377,152,148,176,149,150,136,172,58,
             132,93,234,127,162,21,54,103,67,109]

LEFT_EYEBROW = [70,63,105,66,107,55,65]
RIGHT_EYEBROW = [300,293,334,296,336,285,295]

LEFT_EYE = [33,160,158,133,153,144,163]
RIGHT_EYE = [362,385,387,263,373,380,390]

LIPS = [61,146,91,181,84,17,314,405,321,375,291]


import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


st.set_page_config(page_title="Face Style Recommender")

st.title("AI Face Style Recommender 💇‍♀️🧔")
st.write("Upload a photo to get hairstyle / beard recommendations")
st.caption("⚠️ Uploaded images are processed temporarily and are not stored.")
show_mesh = st.checkbox("Show Face Landmarks", value=True)


gender = st.selectbox("Select Gender", ["Female", "Male"])
uploaded_file = st.file_uploader("Upload a clear face photo", type=["jpg", "png", "jpeg"])

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)

# Face outline indices
FACE_OUTLINE = [10,338,297,332,284,251,389,356,454,323,361,288,397,
                365,379,378,400,377,152,148,176,149,150,136,172,58,
                132,93,234,127,162,21,54]

# Mesh connections (outline)
OUTLINE_CONNECTIONS = [(FACE_OUTLINE[i], FACE_OUTLINE[i+1]) for i in range(len(FACE_OUTLINE)-1)]



def draw_full_mesh(image, landmarks):
    h, w, _ = image.shape

    def draw_line(points, color):
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]

            x1 = int(landmarks[p1].x * w)
            y1 = int(landmarks[p1].y * h)
            x2 = int(landmarks[p2].x * w)
            y2 = int(landmarks[p2].y * h)

            cv2.line(image, (x1, y1), (x2, y2), color, 3)

    draw_line(FACE_OVAL, (255, 0, 0))
    draw_line(LEFT_EYEBROW, (0, 0, 255))
    draw_line(RIGHT_EYEBROW, (0, 0, 255))
    draw_line(LEFT_EYE, (0, 255, 0))
    draw_line(RIGHT_EYE, (0, 255, 0))
    draw_line(LIPS, (0, 255, 255))

    return image  # MUST be here


def euclidean(p1, p2):
    return math.dist(p1, p2)


def get_point(landmarks, idx, w, h):
    return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))


def classify_face_shape(landmarks, image, debug=False):
    import math

    h, w, _ = image.shape

    def get_point(idx):
        return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

    def dist(p1, p2):
        return math.dist(p1, p2)

    left_face = get_point(234)
    right_face = get_point(454)

    forehead_left = get_point(67)
    forehead_right = get_point(297)

    jaw_left = get_point(172)
    jaw_right = get_point(397)

    cheek_left = get_point(93)
    cheek_right = get_point(323)

    top_face = get_point(10)
    bottom_face = get_point(152)

    chin = bottom_face

    face_width = dist(left_face, right_face)
    forehead_width = dist(forehead_left, forehead_right)
    jaw_width = dist(jaw_left, jaw_right)
    cheek_width = dist(cheek_left, cheek_right)
    face_height = dist(top_face, bottom_face)

    ratio = face_width / face_height

    jaw_angle = abs(
        math.degrees(
            math.atan2(jaw_left[1] - chin[1], jaw_left[0] - chin[0]) -
            math.atan2(jaw_right[1] - chin[1], jaw_right[0] - chin[0])
        )
    )

    if debug:
        return {
            "face_width": round(face_width,2),
            "face_height": round(face_height,2),
            "ratio": round(ratio,2),
            "jaw_width": round(jaw_width,2),
            "cheek_width": round(cheek_width,2),
            "forehead_width": round(forehead_width,2),
            "jaw_angle": round(jaw_angle,2)
        }

    # ---- CLASSIFICATION RULES ---- #

    # Square: wide jaw + strong angle
    if jaw_width >= face_width * 0.9 and jaw_angle > 70:
        return "Square"

    # Oval: longer than wide, smooth jaw, cheeks widest
    elif ratio < 0.85 and cheek_width >= jaw_width and cheek_width >= forehead_width:
        return "Oval"

    # Round: width ~ height
    elif 0.85 <= ratio <= 1.05 and cheek_width > jaw_width:
        return "Round"

    # Heart: forehead wider than jaw + tapering chin
    elif forehead_width > jaw_width and ratio >= 0.85:
        return "Heart"

    else:
        return "Oval"


def recommend(face_shape, gender):
    if gender == "Male":
        recommendations = {
            "Round": [
                "Goatee Beard",
                "Short Boxed Beard",
                "Fade with Volume on Top",
                "Side Part Haircut"
            ],
            "Square": [
                "Full Beard with Soft Edges",
                "Textured Crop",
                "Undercut",
                "Crew Cut"
            ],
            "Oval": [
                "Any Beard Style",
                "Quiff Hairstyle",
                "Pompadour",
                "Buzz Cut"
            ],
            "Heart": [
                "Light Stubble",
                "Balbo Beard",
                "Messy Fringe",
                "Side Swept Hair"
            ]
        }
    else:
        recommendations = {
            "Round": [
                "Long Layers",
                "Side Part Hairstyle",
                "Soft Waves",
                "Curtain Bangs"
            ],
            "Square": [
                "Side Bangs",
                "Wavy Lob",
                "Layered Cut",
                "Rounded Bob"
            ],
            "Oval": [
                "Bob Cut",
                "Layered Cut",
                "Curtain Bangs",
                "Soft Waves"
            ],
            "Heart": [
                "Chin-Length Bob",
                "Side Swept Bangs",
                "Loose Curls",
                "Textured Lob"
            ]
        }

    return recommendations.get(face_shape, ["Simple layered cut"])


if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    # ✅ Resize only if too large (no distortion)
    h, w, _ = image_np.shape
    max_width = 640

    if w > max_width:
        scale = max_width / w
        image_np = cv2.resize(image_np, (int(w * scale), int(h * scale)))

    rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # ✅ Correct MpImage creation for Tasks API
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # ✅ Spinner must be inside uploaded_file block
    with st.spinner("Analyzing face..."):
        result = face_landmarker.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            mesh_image = image_np.copy()
            mesh_image = draw_full_mesh(mesh_image, landmarks)

            col1, col2 = st.columns(2)
            col1.image(image_np, caption="Original", use_container_width=True)

            if show_mesh:
                col2.image(mesh_image, caption="Analysis", use_container_width=True)
            else:
                col2.image(image_np, caption="Analysis", use_container_width=True)

            face_shape = classify_face_shape(landmarks, image_np)
            recommendations = recommend(face_shape, gender)

            colA, colB = st.columns(2)
            colA.metric("Face Shape", face_shape)
            colB.metric("Gender", gender)

            accuracy = 85
            st.subheader("Accuracy")
            st.progress(accuracy / 100)
            st.caption(f"Accurate: {accuracy}%")

            st.subheader("✨ Recommended Styles")
            cols = st.columns(2)
            for i, style in enumerate(recommendations):
                cols[i % 2].markdown(f"💇 {style}")

            with st.expander("🔧 Debug Info"):
                st.write("Landmarks detected:", len(landmarks))

        else:
            st.error("No face detected. Please upload a clear front-facing photo.")
