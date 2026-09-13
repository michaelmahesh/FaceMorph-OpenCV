import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path

from src.face_morph import (
    detect_landmarks,
    align_face,
    add_boundary_points,
    calculate_delaunay,
    morph_image,
)


st.set_page_config(
    page_title="FaceMorph-OpenCV",
    page_icon="🔄",
    layout="wide",
)


st.title("🔄 FaceMorph-OpenCV")

st.markdown(
    """
    ### Face Morphing using OpenCV and MediaPipe Face Mesh

    Upload two face images and generate a smooth intermediate face
    using **468 facial landmarks**, alignment, Delaunay triangulation,
    affine warping, color matching, and blending.
    """
)


# ------------------------------------------------------------
# Upload images
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    uploaded_face1 = st.file_uploader(
        "Upload Face 1",
        type=["jpg", "jpeg", "png"],
        key="face1",
    )

with col2:
    uploaded_face2 = st.file_uploader(
        "Upload Face 2",
        type=["jpg", "jpeg", "png"],
        key="face2",
    )


# ------------------------------------------------------------
# Morph strength
# ------------------------------------------------------------

morph_strength = st.slider(
    "Morph Strength",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
)

st.write(f"Selected morph strength: **{morph_strength}%**")


# ------------------------------------------------------------
# Generate button
# ------------------------------------------------------------

if uploaded_face1 is not None and uploaded_face2 is not None:

    if st.button("🚀 Generate Face Morph", type="primary"):

        with st.spinner("Processing face morph..."):

            try:

                # Read uploaded images
                bytes1 = uploaded_face1.getvalue()
                bytes2 = uploaded_face2.getvalue()

                image1 = cv2.imdecode(
                    np.frombuffer(bytes1, np.uint8),
                    cv2.IMREAD_COLOR,
                )

                image2 = cv2.imdecode(
                    np.frombuffer(bytes2, np.uint8),
                    cv2.IMREAD_COLOR,
                )

                if image1 is None:
                    st.error("Unable to read Face 1.")
                    st.stop()

                if image2 is None:
                    st.error("Unable to read Face 2.")
                    st.stop()

                # ------------------------------------------------
                # Working canvas
                # ------------------------------------------------

                height, width = image1.shape[:2]

                image2 = cv2.resize(
                    image2,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )

                # ------------------------------------------------
                # Detect landmarks
                # ------------------------------------------------

                points1 = detect_landmarks(image1)
                points2 = detect_landmarks(image2)

                st.success(
                    f"Detected {len(points1)} landmarks in Face 1 "
                    f"and {len(points2)} landmarks in Face 2."
                )

                # ------------------------------------------------
                # Align Face 2
                # ------------------------------------------------

                aligned_face2, aligned_points2, matrix = align_face(
                    image2,
                    points2,
                    points1,
                    (width, height),
                )

                # ------------------------------------------------
                # Boundary points
                # ------------------------------------------------

                points1_boundary = add_boundary_points(
                    points1,
                    width,
                    height,
                )

                points2_boundary = add_boundary_points(
                    aligned_points2,
                    width,
                    height,
                )

                # ------------------------------------------------
                # Delaunay triangulation
                # ------------------------------------------------

                triangles = calculate_delaunay(
                    points1_boundary,
                    width,
                    height,
                )

                # ------------------------------------------------
                # Target points based on morph strength
                # ------------------------------------------------

                alpha = morph_strength / 100.0

                target_points = (
                    (1.0 - alpha) * points1_boundary
                    + alpha * points2_boundary
                )

                # ------------------------------------------------
                # Warp both faces
                # ------------------------------------------------

                warped_face1 = morph_image(
                    image1,
                    points1_boundary,
                    target_points,
                    triangles,
                )

                warped_face2 = morph_image(
                    aligned_face2,
                    points2_boundary,
                    target_points,
                    triangles,
                )

                # ------------------------------------------------
                # Blend
                # ------------------------------------------------

                result = cv2.addWeighted(
                    warped_face1,
                    1.0 - alpha,
                    warped_face2,
                    alpha,
                    0,
                )

                result_rgb = cv2.cvtColor(
                    result,
                    cv2.COLOR_BGR2RGB,
                )

                # ------------------------------------------------
                # Display
                # ------------------------------------------------

                st.subheader("Face Morph Result")

                st.image(
                    result_rgb,
                    caption=f"{morph_strength}% Morph",
                )

                # ------------------------------------------------
                # Download
                # ------------------------------------------------

                success, encoded = cv2.imencode(
                    ".jpg",
                    result,
                )

                if success:

                    st.download_button(
                        label="⬇️ Download Morph Image",
                        data=encoded.tobytes(),
                        file_name=f"face_morph_{morph_strength}.jpg",
                        mime="image/jpeg",
                    )

                st.info(
                    f"Delaunay triangles used: {len(triangles)}"
                )

            except Exception as e:

                st.error(
                    f"Face morphing failed: {str(e)}"
                )

else:

    st.info(
        "Please upload Face 1 and Face 2 to begin."
    )


st.markdown("---")

st.caption(
    "FaceMorph-OpenCV | OpenCV + MediaPipe Face Mesh"
)