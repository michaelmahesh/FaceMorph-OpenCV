# FaceMorph-OpenCV

Face Morphing using OpenCV and MediaPipe Face Mesh.

This project creates a smooth visual morph between two face images using facial landmarks, automatic alignment, Delaunay triangulation, affine triangle warping, color matching, and facial-mask blending.

## Features

- 468-point MediaPipe Face Mesh landmark detection
- Automatic face alignment
- Delaunhead -20 README.md
ay triangulation
- Triangle-by-triangle affine warping
- Facial-region masking
- Color matching between input faces
- Multiple morph strengths:
  - 25%
  - 50%
  - 75%
- OpenCV-based image processing
- Python 3.11 support

## Processing Pipeline

```text
Face 1 + Face 2
       |
       v
Face Landmark Detection
       |
       v
Automatic Face Alignment
       |
       v
Boundary Point Generation
       |
       v
Delaunay Triangulation
       |
       v
Affine Triangle Warping
       |
       v
Face Mask Generation
       |
       v
Color Matching
       |
       v
Morph Blending
       |
       v
25% / 50% / 75% Results\



python -c 'from pathlib import Path; p=Path("README.md"); p.write_text("""PASTE_COMPLETE_README_HERE""")'
