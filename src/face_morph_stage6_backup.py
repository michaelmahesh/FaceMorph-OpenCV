import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# ============================================================
# FACE MORPH - STAGE 6
#
# Automatic Alignment
# Delaunay Triangulation
# Affine Triangle Warping
# Face Mask
# Color Matching
# Multiple Morph Strengths
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"

IMAGE1 = INPUT_DIR / "face1.jpg"
IMAGE2 = INPUT_DIR / "face2.jpg"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh


# ============================================================
# DETECT LANDMARKS
# ============================================================

def detect_landmarks(image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    ) as face_mesh:

        result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:

        raise RuntimeError(
            "No face detected."
        )

    height, width = image.shape[:2]

    points = []

    for landmark in result.multi_face_landmarks[0].landmark:

        x = landmark.x * width
        y = landmark.y * height

        points.append(
            [x, y]
        )

    return np.array(
        points,
        dtype=np.float32
    )


# ============================================================
# ALIGN FACE 2 TO FACE 1
# ============================================================

def align_face(
    image,
    source_points,
    target_points,
    output_size
):

    # Stable facial landmarks
    alignment_indices = [
        33,     # left eye
        263,    # right eye
        1,      # nose
        61,     # left mouth
        291     # right mouth
    ]

    source = source_points[
        alignment_indices
    ]

    target = target_points[
        alignment_indices
    ]

    matrix, _ = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0
    )

    if matrix is None:

        raise RuntimeError(
            "Face alignment failed."
        )

    width, height = output_size

    aligned_image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Transform all landmarks
    ones = np.ones(
        (source_points.shape[0], 1),
        dtype=np.float32
    )

    homogeneous = np.hstack(
        [
            source_points,
            ones
        ]
    )

    aligned_points = (
        homogeneous @ matrix.T
    )

    aligned_points = aligned_points.astype(
        np.float32
    )

    return (
        aligned_image,
        aligned_points,
        matrix
    )


# ============================================================
# ADD IMAGE BOUNDARY POINTS
# ============================================================

def add_boundary_points(
    points,
    width,
    height
):

    boundary = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],

            [width // 2, 0],
            [width - 1, height // 2],
            [width // 2, height - 1],
            [0, height // 2]
        ],
        dtype=np.float32
    )

    return np.vstack(
        [
            points,
            boundary
        ]
    )


# ============================================================
# DELAUNAY TRIANGULATION
# ============================================================

def calculate_delaunay(
    points,
    width,
    height
):

    subdiv = cv2.Subdiv2D(
        (0, 0, width, height)
    )

    for point in points:

        x = float(point[0])
        y = float(point[1])

        x = max(
            0.0,
            min(x, width - 1.0)
        )

        y = max(
            0.0,
            min(y, height - 1.0)
        )

        subdiv.insert(
            (x, y)
        )

    triangle_list = subdiv.getTriangleList()

    triangles = []

    for triangle in triangle_list:

        x1, y1, x2, y2, x3, y3 = triangle

        triangle_points = np.array(
            [
                [x1, y1],
                [x2, y2],
                [x3, y3]
            ],
            dtype=np.float32
        )

        inside = np.all(
            (triangle_points[:, 0] >= 0)
            &
            (triangle_points[:, 0] < width)
            &
            (triangle_points[:, 1] >= 0)
            &
            (triangle_points[:, 1] < height)
        )

        if not inside:

            continue

        indices = []

        for point in triangle_points:

            distances = np.linalg.norm(
                points - point,
                axis=1
            )

            index = np.argmin(
                distances
            )

            if distances[index] < 1.0:

                indices.append(
                    index
                )

        if len(indices) == 3:

            if len(set(indices)) == 3:

                key = tuple(
                    sorted(indices)
                )

                if key not in triangles:

                    triangles.append(
                        key
                    )

    return triangles


# ============================================================
# WARP ONE TRIANGLE
# ============================================================

def warp_triangle(
    image,
    output,
    source_triangle,
    destination_triangle
):

    source_triangle = np.float32(
        source_triangle
    )

    destination_triangle = np.float32(
        destination_triangle
    )

    source_rect = cv2.boundingRect(
        source_triangle
    )

    destination_rect = cv2.boundingRect(
        destination_triangle
    )

    sx, sy, sw, sh = source_rect
    dx, dy, dw, dh = destination_rect

    if sw <= 0 or sh <= 0:

        return

    if dw <= 0 or dh <= 0:

        return

    source_local = (
        source_triangle
        -
        np.array(
            [sx, sy],
            dtype=np.float32
        )
    )

    destination_local = (
        destination_triangle
        -
        np.array(
            [dx, dy],
            dtype=np.float32
        )
    )

    matrix = cv2.getAffineTransform(
        source_local,
        destination_local
    )

    warped = cv2.warpAffine(
        image[
            sy:sy + sh,
            sx:sx + sw
        ],
        matrix,
        (dw, dh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    mask = np.zeros(
        (dh, dw),
        dtype=np.float32
    )

    cv2.fillConvexPoly(
        mask,
        np.int32(destination_local),
        1.0,
        lineType=cv2.LINE_AA
    )

    mask = np.dstack(
        [
            mask,
            mask,
            mask
        ]
    )

    destination_region = output[
        dy:dy + dh,
        dx:dx + dw
    ]

    warped = warped.astype(
        np.float32
    )

    output[
        dy:dy + dh,
        dx:dx + dw
    ] = (
        destination_region
        *
        (1.0 - mask)
        +
        warped
        *
        mask
    )


# ============================================================
# WARP COMPLETE IMAGE
# ============================================================

def morph_image(
    image,
    source_points,
    destination_points,
    triangles
):

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    for triangle in triangles:

        source_triangle = [
            source_points[i]
            for i in triangle
        ]

        destination_triangle = [
            destination_points[i]
            for i in triangle
        ]

        warp_triangle(
            image,
            output,
            source_triangle,
            destination_triangle
        )

    return np.clip(
        output,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# FACE MASK
# ============================================================

def create_face_mask(
    points,
    width,
    height
):

    # Use facial landmarks only
    face_points = np.int32(
        points[:468]
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # Convex hull
    hull = cv2.convexHull(
        face_points
    )

    cv2.fillConvexPoly(
        mask,
        hull,
        255
    )

    # Slight erosion prevents the morph
    # from reaching too far outside the face
    kernel = np.ones(
        (9, 9),
        dtype=np.uint8
    )

    mask = cv2.erode(
        mask,
        kernel,
        iterations=1
    )

    # Feather boundary
    mask = cv2.GaussianBlur(
        mask,
        (51, 51),
        0
    )

    mask = (
        mask.astype(
            np.float32
        )
        /
        255.0
    )

    mask = np.dstack(
        [
            mask,
            mask,
            mask
        ]
    )

    return mask


# ============================================================
# COLOR MATCHING
# ============================================================

def match_color(
    source,
    target,
    mask
):
    """
    Adjust source colors to approximately match target.
    """

    source_float = source.astype(
        np.float32
    )

    target_float = target.astype(
        np.float32
    )

    # Use the mask to calculate facial statistics
    valid = mask[:, :, 0] > 0.5

    if np.count_nonzero(valid) < 100:

        return source

    source_pixels = source_float[
        valid
    ]

    target_pixels = target_float[
        valid
    ]

    source_mean = np.mean(
        source_pixels,
        axis=0
    )

    target_mean = np.mean(
        target_pixels,
        axis=0
    )

    source_std = np.std(
        source_pixels,
        axis=0
    )

    target_std = np.std(
        target_pixels,
        axis=0
    )

    source_std = np.maximum(
        source_std,
        1.0
    )

    # Normalize source
    adjusted = (
        source_float
        -
        source_mean
    )

    adjusted = (
        adjusted
        /
        source_std
    )

    adjusted = (
        adjusted
        *
        target_std
    )

    adjusted = (
        adjusted
        +
        target_mean
    )

    return np.clip(
        adjusted,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    path,
    image
):

    cv2.imwrite(
        str(path),
        image
    )

    print(
        f"Saved: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FACE MORPH - STAGE 6")
    print("ALIGNMENT + COLOR MATCHING + MORPH STRENGTH")
    print("=" * 70)

    # ========================================================
    # LOAD
    # ========================================================

    print("\nLoading images...")

    image1 = cv2.imread(
        str(IMAGE1)
    )

    image2_original = cv2.imread(
        str(IMAGE2)
    )

    if image1 is None:

        raise RuntimeError(
            "Could not load input/face1.jpg"
        )

    if image2_original is None:

        raise RuntimeError(
            "Could not load input/face2.jpg"
        )

    height, width = image1.shape[:2]

    print(
        f"Face 1: {image1.shape}"
    )

    print(
        f"Face 2: {image2_original.shape}"
    )

    # ========================================================
    # RESIZE FACE 2
    # ========================================================

    image2 = cv2.resize(
        image2_original,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    print(
        f"Working canvas: "
        f"{width} x {height}"
    )

    # ========================================================
    # LANDMARKS
    # ========================================================

    print(
        "\nDetecting Face 1 landmarks..."
    )

    points1 = detect_landmarks(
        image1
    )

    print(
        f"Face 1 landmarks: "
        f"{len(points1)}"
    )

    print(
        "\nDetecting Face 2 landmarks..."
    )

    points2 = detect_landmarks(
        image2
    )

    print(
        f"Face 2 landmarks: "
        f"{len(points2)}"
    )

    # ========================================================
    # ALIGN FACE 2
    # ========================================================

    print(
        "\nAligning Face 2..."
    )

    image2_aligned, points2_aligned, matrix = align_face(
        image2,
        points2,
        points1,
        (width, height)
    )

    print(
        "Face alignment completed."
    )

    print(
        "\nAlignment matrix:"
    )

    print(
        matrix
    )

    # ========================================================
    # SAVE ALIGNED IMAGE
    # ========================================================

    save_image(
        OUTPUT_DIR / "face2_aligned_stage6.jpg",
        image2_aligned
    )

    # ========================================================
    # CREATE INITIAL ALIGNMENT PREVIEW
    # ========================================================

    alignment_preview = cv2.addWeighted(
        image1,
        0.5,
        image2_aligned,
        0.5,
        0
    )

    save_image(
        OUTPUT_DIR / "alignment_preview_stage6.jpg",
        alignment_preview
    )

    # ========================================================
    # CREATE FACE MASK
    # ========================================================

    print(
        "\nCreating face mask..."
    )

    # Temporary points with boundary
    points1_full = add_boundary_points(
        points1,
        width,
        height
    )

    points2_full = add_boundary_points(
        points2_aligned,
        width,
        height
    )

    average_points = (
        points1_full
        +
        points2_full
    ) / 2.0

    face_mask = create_face_mask(
        average_points,
        width,
        height
    )

    mask_preview = (
        face_mask[:, :, 0]
        * 255
    ).astype(
        np.uint8
    )

    save_image(
        OUTPUT_DIR / "face_mask_stage6.jpg",
        mask_preview
    )

    # ========================================================
    # COLOR MATCH FACE 2
    # ========================================================

    print(
        "\nMatching Face 2 color to Face 1..."
    )

    image2_color_matched = match_color(
        image2_aligned,
        image1,
        face_mask
    )

    save_image(
        OUTPUT_DIR / "face2_color_matched.jpg",
        image2_color_matched
    )

    # ========================================================
    # DELAUNAY
    # ========================================================

    print(
        "\nCalculating Delaunay triangulation..."
    )

    triangles = calculate_delaunay(
        average_points,
        width,
        height
    )

    print(
        f"Delaunay triangles: "
        f"{len(triangles)}"
    )

    # ========================================================
    # TRIANGULATION PREVIEW
    # ========================================================

    triangulation_preview = cv2.addWeighted(
        image1,
        0.5,
        image2_color_matched,
        0.5,
        0
    )

    for triangle in triangles:

        p1 = np.int32(
            average_points[
                triangle[0]
            ]
        )

        p2 = np.int32(
            average_points[
                triangle[1]
            ]
        )

        p3 = np.int32(
            average_points[
                triangle[2]
            ]
        )

        cv2.line(
            triangulation_preview,
            tuple(p1),
            tuple(p2),
            (0, 255, 0),
            1
        )

        cv2.line(
            triangulation_preview,
            tuple(p2),
            tuple(p3),
            (0, 255, 0),
            1
        )

        cv2.line(
            triangulation_preview,
            tuple(p3),
            tuple(p1),
            (0, 255, 0),
            1
        )

    save_image(
        OUTPUT_DIR / "delaunay_stage6.jpg",
        triangulation_preview
    )

    # ========================================================
    # WARP FACE 1
    # ========================================================

    print(
        "\nWarping Face 1..."
    )

    warped1 = morph_image(
        image1,
        points1_full,
        average_points,
        triangles
    )

    # ========================================================
    # WARP FACE 2
    # ========================================================

    print(
        "Warping Face 2..."
    )

    warped2 = morph_image(
        image2_color_matched,
        points2_full,
        average_points,
        triangles
    )

    # ========================================================
    # SAVE WARPED IMAGES
    # ========================================================

    save_image(
        OUTPUT_DIR / "stage6_warped_face1.jpg",
        warped1
    )

    save_image(
        OUTPUT_DIR / "stage6_warped_face2.jpg",
        warped2
    )

    # ========================================================
    # BLEND DIFFERENT MORPH STRENGTHS
    # ========================================================

    print(
        "\nCreating morph results..."
    )

    warped1_float = warped1.astype(
        np.float32
    )

    warped2_float = warped2.astype(
        np.float32
    )

    background = image1.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Function for creating one result
    # --------------------------------------------------------

    def create_morph(
        face2_strength
    ):

        face1_strength = (
            1.0 -
            face2_strength
        )

        blended_face = (
            warped1_float
            *
            face1_strength
            +
            warped2_float
            *
            face2_strength
        )

        result = (
            blended_face
            *
            face_mask
            +
            background
            *
            (
                1.0 -
                face_mask
            )
        )

        return np.clip(
            result,
            0,
            255
        ).astype(
            np.uint8
        )

    # ========================================================
    # 25% FACE 2
    # ========================================================

    morph25 = create_morph(
        0.25
    )

    save_image(
        OUTPUT_DIR / "face_morph_25.jpg",
        morph25
    )

    # ========================================================
    # 50% FACE 2
    # ========================================================

    morph50 = create_morph(
        0.50
    )

    save_image(
        OUTPUT_DIR / "face_morph_50.jpg",
        morph50
    )

    # ========================================================
    # 75% FACE 2
    # ========================================================

    morph75 = create_morph(
        0.75
    )

    save_image(
        OUTPUT_DIR / "face_morph_75.jpg",
        morph75
    )

    # ========================================================
    # FINAL DEFAULT RESULT
    # ========================================================

    save_image(
        OUTPUT_DIR / "face_morph_stage6.jpg",
        morph50
    )

    # Also update main result
    save_image(
        OUTPUT_DIR / "face_morph.jpg",
        morph50
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STAGE 6 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        "  face2_aligned_stage6.jpg"
    )

    print(
        "  alignment_preview_stage6.jpg"
    )

    print(
        "  face_mask_stage6.jpg"
    )

    print(
        "  face2_color_matched.jpg"
    )

    print(
        "  delaunay_stage6.jpg"
    )

    print(
        "  stage6_warped_face1.jpg"
    )

    print(
        "  stage6_warped_face2.jpg"
    )

    print(
        "  face_morph_25.jpg"
    )

    print(
        "  face_morph_50.jpg"
    )

    print(
        "  face_morph_75.jpg"
    )

    print(
        "  face_morph_stage6.jpg"
    )

    print(
        "  face_morph.jpg"
    )

    print("\n" + "=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()