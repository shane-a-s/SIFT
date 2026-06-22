# SIFT: Scale-Invariant Feature Transform

A from-scratch Python implementation of the SIFT feature-detection pipeline,
built with NumPy and OpenCV, including a side-by-side benchmark against OpenCV's
built-in SIFT detector.

Extension of Machine Vision assignment at Munster Technological University.

---

## Pipeline

| Step                              | What it does                                                                                                                                                                                                                                                                 |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Octave pyramid**          | Doubles the input, then builds*n* octaves each with `s+3` Gaussian-blurred images. Sigma steps by `k = 2^(1/s)` within each octave; the octave seeds from the image where sigma has doubled, downsampled by 2x.                                                        |
| **Difference of Gaussians** | Subtracts adjacent scale-space images within each octave, a fast approximation to the Laplacian of Gaussian.                                                                                                                                                                |
| **Keypoint detection**      | Finds local 3-D extrema (maxima and minima) across scale and space. Candidates must exceed a contrast threshold in absolute value.                                                                                                                                          |
| **Sub-pixel localisation**  | Fits a 3-D Taylor quadratic to each extremum and solves for the true sub-pixel offset via `-H^-1 deltaD`. Iterates up to 5 times, discarding non-converging or low-contrast points.                                                                                        |
| **Edge rejection**          | Computes the 2x2 spatial Hessian ratio `trace^2/det`. Points on edges have a near-zero determinant; they are rejected using Lowe's threshold `r = 10`.                                                                                                                   |
| **Orientation assignment**  | Samples a Gaussian-weighted 7x7 gradient grid (step = 1.5sigma) around each keypoint and builds a 36-bin histogram. The dominant bin gives the keypoint's canonical orientation.                                                                                             |
| **Descriptors**             | Samples a Gaussian-weighted 16x16 gradient grid, rotates angles relative to the dominant orientation, divides into 4x4 sub-grids, and builds an 8-bin histogram per sub-grid using trilinear interpolation. The resulting 128-D vector is L2-normalised and clipped at 0.2. |
| **OpenCV comparison**       | Runs OpenCV's SIFT on the same image and produces a side-by-side keypoint overlay plus a histogram of nearest-neighbour L2 descriptor distances.                                                                                                                             |

All outputs are saved to `outputs/`.

## Requirements

Python 3.8+

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# run on the default image
python sift.py

# run on any image
python sift.py path/to/image.png

# tune detection sensitivity (default 3.5, lower = more keypoints)
python sift.py path/to/image.png --threshold 3.0

# control edge rejection aggressiveness (default 10.0, lower = stricter)
python sift.py path/to/image.png --edge-threshold 10.0

# fix the number of octaves (default: auto from image size)
python sift.py path/to/image.png --octaves 4

# trigger comparison against opencv SIFT implementation
python sift.py path/to/image.png --comparison opencv
```

---
