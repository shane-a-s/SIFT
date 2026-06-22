import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "outputs"

# Lowe (2004) defaults
N_SCALES_PER_OCTAVE = 3   # s - DoG extrema found in the middle s levels of s+2
SIGMA_0 = 1.6             # base blur for the first scale in each octave
SIGMA_INPUT = 0.5         # assumed blur already present in the input from the camera sensor


def save_fig(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


#  Image loading 
def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    # Single-channel float32 for all processing. keeping the colour original for display
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    print(f"Loaded '{path}'  shape={gray.shape}  dtype={gray.dtype}")
    return image, gray


#  Task 1: Octave Pyramid 
def build_octave_pyramid(gray, n_octaves, s=N_SCALES_PER_OCTAVE, sigma_0=SIGMA_0):
    """
    Build a Gaussian scale-space octave pyramid.

    Each octave contains s+3 blurred images, yielding s+2 DoG images.  Extrema are
    found only in the inner s DoG levels so that each candidate has a full 3x3x3
    neighbourhood (one DoG layer above and below in scale).

    The input is first doubled in resolution - the "octave -1" trick
    - so that features at finer scales than the original pixel spacing can be found.
    Each subsequent octave seeds from the image where sigma has just doubled, then
    downsamples by 2, halving the resolution.

    Within an octave, sigmas step by k = 2^(1/s) so that exactly s steps double
    the blur, maintaining consistency across octave boundaries.
    """
    k = 2.0 ** (1.0 / s)
    # s+3 sigmas per octave: sigma_0, sigma_0*k, ..., sigma_0*k^(s+2)
    sigmas = [sigma_0 * (k ** i) for i in range(s + 3)]

    # Double the image - octave 0 therefore operates at 2x the original resolution,
    # capturing features at half the original pixel spacing
    base = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2),
                      interpolation=cv2.INTER_LINEAR)

    # After upsampling 2x, the effective blur is SIGMA_INPUT * 2.
    # Apply the incremental sigma to reach sigma_0.
    assumed = SIGMA_INPUT * 2.0
    if sigma_0 > assumed:
        base = cv2.GaussianBlur(base, (0, 0), np.sqrt(sigma_0 ** 2 - assumed ** 2))

    octaves = []
    for _ in range(n_octaves):
        imgs = [base]
        for i in range(1, s + 3):
            # Incremental blur: compose two Gaussians analytically instead of
            # blurring the original each time - avoids accumulated rounding error
            prev_sigma = sigma_0 * (k ** (i - 1))
            curr_sigma = sigma_0 * (k ** i)
            inc = np.sqrt(curr_sigma ** 2 - prev_sigma ** 2)
            imgs.append(cv2.GaussianBlur(imgs[-1], (0, 0), inc))
        octaves.append(imgs)
        # The image at index s has sigma = sigma_0 * 2 (one full doubling).
        # Downsample it by 2 to seed the next octave at the same effective scale.
        seed = imgs[s]
        base = cv2.resize(seed, (seed.shape[1] // 2, seed.shape[0] // 2),
                          interpolation=cv2.INTER_NEAREST)

    return octaves, sigmas


def compute_dog_pyramid(octaves):
    """Subtract adjacent scale-space images within each octave to get DoG images."""
    return [[oct[i + 1] - oct[i] for i in range(len(oct) - 1)] for oct in octaves]


def octave_to_orig(x, y, sigma, octave_idx):
    """
    Convert a point from octave-local coordinates to original image space.

    Octave 0 is at 2x resolution, so its coordinates are divided by 2.
    Octave 1 is at 1x (original), octave 2 at 0.5x, etc.
    The coordinate scale factor is therefore 2^(octave_idx - 1).
    """
    scale = 2.0 ** (octave_idx - 1)
    return x * scale, y * scale, sigma * scale


def visualise_pyramid(octaves, dog_octaves):
    """Show the base image and a representative DoG for each octave."""
    n = len(octaves)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes[:, np.newaxis]
    for i in range(n):
        h, w = octaves[i][0].shape
        axes[0, i].imshow(octaves[i][0], cmap="gray")
        axes[0, i].set_title(f"Octave {i}\n{w}x{h} px")
        axes[0, i].axis("off")
        axes[1, i].imshow(dog_octaves[i][1], cmap="gray")
        axes[1, i].set_title("DoG (scale 1)")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Scale space", fontsize=10)
    axes[1, 0].set_ylabel("DoG", fontsize=10)
    fig.suptitle("Octave Pyramid - Base Scale and DoG per Octave")
    plt.tight_layout()
    save_fig(fig, "1_octave_pyramid.png")


#  Task 2: Keypoint Detection 

def detect_keypoints_pyramid(dog_octaves, sigmas_per_octave, threshold=10.0):
    """
    Find local extrema (maxima and minima) across all octaves and all scales.

    A candidate is accepted if |DoG| exceeds the contrast threshold AND the value
    is strictly greater than, or strictly less than, all 26 neighbours in the 3x3x3
    (x, y, scale) neighbourhood.

    Keypoints are returned with both their octave-local coordinates (needed for
    refinement and descriptor sampling) and their original-image coordinates
    (needed for visualisation and comparison with OpenCV).
    """
    keypoints = []
    for oct_idx, dog_images in enumerate(dog_octaves):
        dog_stack = np.stack(dog_images, axis=0)
        S, H, W = dog_stack.shape
        for s in range(1, S - 1):
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    value = dog_stack[s, y, x]
                    if abs(value) < threshold:
                        continue
                    nb = dog_stack[s-1:s+2, y-1:y+2, x-1:x+2].copy()
                    nb[1, 1, 1] = np.inf
                    is_min = value < np.min(nb)
                    nb[1, 1, 1] = -np.inf
                    is_max = value > np.max(nb)
                    if is_min or is_max:
                        ox, oy, osigma = octave_to_orig(
                            float(x), float(y), sigmas_per_octave[s], oct_idx)
                        keypoints.append({
                            "x": float(x), "y": float(y),
                            "sigma": sigmas_per_octave[s],
                            "octave": oct_idx, "scale": s,
                            "x_orig": ox, "y_orig": oy, "sigma_orig": osigma,
                        })
    print(f"  Initial detection: {len(keypoints)} keypoints")
    return keypoints


#  Sub-pixel Localisation

def refine_keypoints(keypoints, dog_octaves, sigmas_per_octave,
                     contrast_threshold=10.0, max_iter=5):
    """
    Improve keypoint accuracy by fitting a 3-D quadratic to the DoG and finding
    the true sub-pixel extremum via the Taylor expansion:

        D(x + delta) ≈ D(x) + (∂D/∂x)ᵀ delta + ½ deltaᵀ H delta

    Setting ∂D/∂(delta) = 0 gives:  delta = -H^-1 (∂D/∂x)

    If |delta| > 0.5 in any dimension the true peak lies in a neighbouring voxel;
    this moves there and iterate.  Points that do not converge, fall outside the
    image border, or whose interpolated DoG value remains below the contrast
    threshold are discarded.
    """
    refined = []
    for kp in keypoints:
        oct_idx = kp["octave"]
        dog_stack = np.stack(dog_octaves[oct_idx], axis=0)
        S, H, W = dog_stack.shape

        s = kp["scale"]
        y = int(round(kp["y"]))
        x = int(round(kp["x"]))
        converged = False
        delta = np.zeros(3)
        grad = np.zeros(3)

        for _ in range(max_iter):
            if not (1 <= s < S-1 and 1 <= y < H-1 and 1 <= x < W-1):
                break
            D = dog_stack[s-1:s+2, y-1:y+2, x-1:x+2]

            # First-order partial derivatives (central differences)
            grad = np.array([
                (D[2,1,1] - D[0,1,1]) / 2.0,  # ∂D/∂s
                (D[1,2,1] - D[1,0,1]) / 2.0,  # ∂D/∂y
                (D[1,1,2] - D[1,1,0]) / 2.0,  # ∂D/∂x
            ])

            # Second-order partial derivatives (Hessian)
            dss = D[2,1,1] - 2*D[1,1,1] + D[0,1,1]
            dyy = D[1,2,1] - 2*D[1,1,1] + D[1,0,1]
            dxx = D[1,1,2] - 2*D[1,1,1] + D[1,1,0]
            dsy = (D[2,2,1] - D[2,0,1] - D[0,2,1] + D[0,0,1]) / 4.0
            dsx = (D[2,1,2] - D[2,1,0] - D[0,1,2] + D[0,1,0]) / 4.0
            dyx = (D[1,2,2] - D[1,2,0] - D[1,0,2] + D[1,0,0]) / 4.0
            H_mat = np.array([[dss, dsy, dsx],
                              [dsy, dyy, dyx],
                              [dsx, dyx, dxx]])
            try:
                delta = -np.linalg.solve(H_mat, grad)
            except np.linalg.LinAlgError:
                break

            if np.max(np.abs(delta)) < 0.5:
                converged = True
                break

            # Shift to the neighbouring voxel the peak has moved into
            s += int(round(delta[0]))
            y += int(round(delta[1]))
            x += int(round(delta[2]))

        if not converged:
            continue
        if not (1 <= s < S-1 and 1 <= y < H-1 and 1 <= x < W-1):
            continue

        # Interpolated DoG value at the refined location
        D_interp = dog_stack[s, y, x] + 0.5 * grad @ delta
        if abs(D_interp) < contrast_threshold:
            continue

        rx, ry = x + delta[2], y + delta[1]
        rsigma = sigmas_per_octave[s]
        ox, oy, osigma = octave_to_orig(rx, ry, rsigma, oct_idx)
        refined.append({
            "x": rx, "y": ry, "sigma": rsigma,
            "octave": oct_idx, "scale": s,
            "x_orig": ox, "y_orig": oy, "sigma_orig": osigma,
        })

    print(f"  Sub-pixel refinement: {len(keypoints)} -> {len(refined)} keypoints")
    return refined


#  Edge Response Rejection 

def reject_edge_keypoints(keypoints, dog_octaves, edge_threshold=10.0):
    """
    Reject keypoints that lie on edges rather than corners.

    An edge has one large principal curvature and one near-zero, making the
    spatial Hessian near-singular.  measure this with the ratio:

        r = trace(H)^2 / det(H)  =  (Dxx + Dyy)^2 / (Dxx·Dyy - Dxy^2)

    A perfect corner has equal curvatures: r = 4.
    A pure edge has det -> 0: r -> inf.
    Lowe uses r_max = 10, rejecting points where r > (r_max + 1)^2 / r_max = 12.1.
    """
    ratio_threshold = (edge_threshold + 1.0) ** 2 / edge_threshold
    kept = []
    for kp in keypoints:
        D = dog_octaves[kp["octave"]][kp["scale"]]
        H_img, W_img = D.shape
        xi, yi = int(round(kp["x"])), int(round(kp["y"]))
        if not (1 <= yi < H_img-1 and 1 <= xi < W_img-1):
            kept.append(kp)
            continue
        dxx = D[yi, xi+1] - 2*D[yi, xi] + D[yi, xi-1]
        dyy = D[yi+1, xi] - 2*D[yi, xi] + D[yi-1, xi]
        dxy = (D[yi+1, xi+1] - D[yi+1, xi-1] - D[yi-1, xi+1] + D[yi-1, xi-1]) / 4.0
        det = dxx * dyy - dxy * dxy
        # det <= 0 is a saddle point - not a true extremum
        if det <= 0:
            continue
        if (dxx + dyy) ** 2 / det < ratio_threshold:
            kept.append(kp)
    print(f"  Edge rejection: {len(keypoints)} -> {len(kept)} keypoints")
    return kept


def visualise_keypoints(image, keypoints, fname, title):
    """Draw keypoints in original image coordinates."""
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    for kp in keypoints:
        cx, cy = int(round(kp["x_orig"])), int(round(kp["y_orig"]))
        r = max(1, int(round(kp["sigma_orig"])))
        cv2.circle(overlay, (cx, cy), r, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    save_fig(fig, fname)


#  Task 3: Gradients and Orientation 

def compute_gradients_pyramid(octaves):
    """
    Compute x and y gradients for every scale in every octave using a
    [1, 0, -1] finite-difference kernel applied to the already-blurred images.
    Computing gradients at the octave's own resolution means the gradient scale
    matches the scale at which each keypoint was detected.
    """
    # [-1, 0, 1] gives src[c+1] - src[c-1] - the standard forward gradient.
    # [1, 0, -1] would negate both components, rotating every angle by 180 degrees.
    dx_k = np.array([[-1, 0, 1]], dtype=np.float32)
    dy_k = dx_k.T
    gx_oct = [[cv2.filter2D(img, -1, dx_k) for img in oct] for oct in octaves]
    gy_oct = [[cv2.filter2D(img, -1, dy_k) for img in oct] for oct in octaves]
    return gx_oct, gy_oct


def assign_orientations(keypoints, gx_octaves, gy_octaves, sigmas_per_octave):
    """
    Assign a dominant orientation to each keypoint using a 36-bin gradient histogram.

    Sampling and weighting are done entirely within the keypoint's own octave so
    that spatial distances are measured in the coordinate system where the keypoint
    was detected - keeping everything scale-consistent.

    Steps per keypoint:
      1. Sample a 7x7 grid of gradients spaced by 1.5sigma around the keypoint.
      2. Weight each sample by a Gaussian (sigma_weight = 1.5 x keypoint sigma).
      3. Accumulate weighted magnitudes into a 36-bin histogram (10 degrees per bin).
      4. The dominant orientation is the centre angle of the peak bin.
    """
    offset = np.arange(-3, 4, dtype=np.float32)
    oriented = []

    for kp in keypoints:
        oct_idx, s_idx = kp["octave"], kp["scale"]
        x, y, sigma = kp["x"], kp["y"], kp["sigma"]
        gx_img = gx_octaves[oct_idx][s_idx]
        gy_img = gy_octaves[oct_idx][s_idx]
        H, W = gx_img.shape

        m_grid = np.zeros((7, 7), dtype=np.float32)
        theta_grid = np.zeros((7, 7), dtype=np.float32)
        weights = np.zeros((7, 7), dtype=np.float32)

        for ri, ky in enumerate(offset):
            for ci, kx in enumerate(offset):
                # Scale-aware spacing: step = 1.5sigma
                ox, oy = 1.5 * kx * sigma, 1.5 * ky * sigma
                xs = int(np.clip(np.round(x + ox), 0, W - 1))
                ys = int(np.clip(np.round(y + oy), 0, H - 1))
                gxv, gyv = gx_img[ys, xs], gy_img[ys, xs]
                m_grid[ri, ci] = np.sqrt(gxv*gxv + gyv*gyv)
                theta_grid[ri, ci] = np.arctan2(gyv, gxv)
                weights[ri, ci] = (
                    np.exp(-(ox*ox + oy*oy) / ((9.0*sigma*sigma) / 2.0))
                    / ((9.0*np.pi*sigma*sigma) / 2.0)
                )

        # 36-bin histogram; arctan2 elemof [−pi, pi] mapped to bins [0, 35]
        hist = np.zeros(36, dtype=np.float32)
        for ri in range(7):
            for ci in range(7):
                bin_i = int(np.clip(
                    np.floor(36.0 * theta_grid[ri, ci] / (2.0*np.pi)), -18, 17)) + 18
                hist[bin_i] += weights[ri, ci] * m_grid[ri, ci]

        peak = np.argmax(hist)
        theta_hat = (2.0*np.pi / 36.0) * ((peak - 18) + 0.5)
        oriented.append({**kp, "theta_hat": theta_hat})

    return oriented


def visualise_oriented(image, oriented_kps, fname, title):
    """Draw oriented keypoints (circle + orientation line) in original image space."""
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    for kp in oriented_kps:
        cx, cy = int(round(kp["x_orig"])), int(round(kp["y_orig"]))
        r = max(1, int(round(kp["sigma_orig"])))
        theta = kp["theta_hat"]
        ex = int(round(cx + r * np.cos(theta)))
        ey = int(round(cy + r * np.sin(theta)))
        cv2.circle(overlay, (cx, cy), r, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (cx, cy), (ex, ey), (0, 0, 255), 1, lineType=cv2.LINE_AA)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    save_fig(fig, fname)


#  Task 4: SIFT Descriptors 

def build_descriptors(oriented_kps, gx_octaves, gy_octaves):
    """
    Build a 128-D SIFT descriptor for each oriented keypoint.

    Sampling is done within the keypoint's own octave at the keypoint's scale,
    so gradient resolution matches detection resolution.

    Steps per keypoint:
      1. Sample a 16x16 grid of gradients spaced by (9/16)sigma.
      2. Subtract the dominant orientation -> rotation-invariant relative angles.
      3. Divide the 16x16 grid into 16 non-overlapping 4x4 sub-grids.
      4. Within each sub-grid build an 8-bin orientation histogram using trilinear
         interpolation - each sample contributes to the two nearest spatial cells
         in each dimension and the two nearest orientation bins, weighted by distance.
         This avoids discontinuities when a sample lies near a cell or bin boundary.
      5. Concatenate the 16x8 = 128 values into one vector.
      6. L2-normalise; clip at 0.2 to limit the influence of dominant gradients.
    """
    offset_vals = np.arange(-8, 8, dtype=np.float32)
    descriptors = []

    for kp in oriented_kps:
        oct_idx, s_idx = kp["octave"], kp["scale"]
        x, y, sigma = kp["x"], kp["y"], kp["sigma"]
        theta_hat = kp["theta_hat"]
        gx_img = gx_octaves[oct_idx][s_idx]
        gy_img = gy_octaves[oct_idx][s_idx]
        H, W = gx_img.shape

        # Scale-aware spacing: 16 steps cover 9sigma pixels on each side
        offsets = (9.0 / 16.0) * (offset_vals + 0.5) * sigma
        descriptor = np.zeros((4, 4, 8), dtype=np.float32)

        for ri, t in enumerate(offsets):
            for ci, s in enumerate(offsets):
                xi = int(np.clip(np.round(x + s), 0, W - 1))
                yi = int(np.clip(np.round(y + t), 0, H - 1))
                gxv, gyv = gx_img[yi, xi], gy_img[yi, xi]

                # Gaussian weight: de-emphasises samples far from the keypoint centre
                w = (np.exp(-(s*s + t*t) / ((81.0*sigma*sigma) / 2.0))
                     / ((81.0*np.pi*sigma*sigma) / 2.0))
                magnitude = np.sqrt(gxv*gxv + gyv*gyv)

                # Relative angle: subtract dominant orientation and wrap to [0, 2pi)
                theta_rel = (np.arctan2(gyv, gxv) - theta_hat) % (2*np.pi)

                # Trilinear interpolation 
                # Split the contribution between the 2 nearest spatial cells in
                # each dimension (row, col) and the 2 nearest orientation bins,
                # weighted by fractional distance to each boundary.
                frac_row = ri / 4.0
                frac_col = ci / 4.0
                cell_r = int(frac_row)
                cell_c = int(frac_col)
                dr = frac_row - cell_r   # weight toward the next row cell
                dc = frac_col - cell_c   # weight toward the next col cell

                frac_bin = (8.0 * theta_rel) / (2.0 * np.pi)
                bin0 = int(frac_bin) % 8
                bin1 = (bin0 + 1) % 8
                db = frac_bin - int(frac_bin)  # weight toward the next bin

                contrib = w * magnitude
                for dr_off, dr_w in [(0, 1.0 - dr), (1, dr)]:
                    r_idx = cell_r + dr_off
                    if r_idx >= 4:
                        continue
                    for dc_off, dc_w in [(0, 1.0 - dc), (1, dc)]:
                        c_idx = cell_c + dc_off
                        if c_idx >= 4:
                            continue
                        cw = contrib * dr_w * dc_w
                        descriptor[r_idx, c_idx, bin0] += cw * (1.0 - db)
                        descriptor[r_idx, c_idx, bin1] += cw * db

        desc_vec = descriptor.flatten()
        norm = np.sqrt(np.sum(desc_vec * desc_vec))
        if norm > 0:
            desc_vec /= norm
        # Clip at 0.2 to reduce the effect of any single dominant gradient direction
        desc_vec = np.clip(desc_vec, 0.0, 0.2)
        descriptors.append({**kp, "descriptor": desc_vec})

    return descriptors


#  OpenCV Benchmark Comparison

def compare_with_opencv(image, gray, this_oriented_kps, this_descriptors,
                        contrast_threshold=0.04, edge_threshold=10):
    """
    Side-by-side comparison against OpenCV's built-in SIFT.

    OpenCV SIFT is a complete, optimised implementation of Lowe (2004).  Comparing
    against it shows how close this implementation is and where the remaining gaps are.

    Two outputs are produced:
      - Keypoint overlay (side by side)
      - Histogram of L2 distances from each of these descriptors to the nearest
        OpenCV descriptor (0 = identical, sqrt(2) =~ 1.41 = orthogonal vectors)
    """
    sift = cv2.SIFT_create(contrastThreshold=contrast_threshold,
                           edgeThreshold=edge_threshold)
    cv_kps, cv_descs = sift.detectAndCompute(gray.astype(np.uint8), None)
    print(f"\n  OpenCV SIFT: {len(cv_kps)} keypoints")
    print(f"  This SIFT:    {len(this_oriented_kps)} keypoints")

    this_overlay = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    cv_overlay  = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    for kp in this_oriented_kps:
        cx, cy = int(round(kp["x_orig"])), int(round(kp["y_orig"]))
        r = max(1, int(round(kp["sigma_orig"])))
        theta = kp["theta_hat"]
        ex = int(round(cx + r * np.cos(theta)))
        ey = int(round(cy + r * np.sin(theta)))
        cv2.circle(this_overlay, (cx, cy), r, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.line(this_overlay, (cx, cy), (ex, ey), (0, 0, 255), 1, lineType=cv2.LINE_AA)

    for kp in cv_kps:
        cx, cy = int(round(kp.pt[0])), int(round(kp.pt[1]))
        r = max(1, int(round(kp.size / 2)))
        angle_rad = np.deg2rad(kp.angle)
        ex = int(round(cx + r * np.cos(angle_rad)))
        ey = int(round(cy + r * np.sin(angle_rad)))
        cv2.circle(cv_overlay, (cx, cy), r, (255, 100, 0), 1, lineType=cv2.LINE_AA)
        cv2.line(cv_overlay, (cx, cy), (ex, ey), (255, 0, 0), 1, lineType=cv2.LINE_AA)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    axes[0].imshow(this_overlay)
    axes[0].set_title(f"This SIFT\n{len(this_oriented_kps)} keypoints", fontsize=13)
    axes[0].axis("off")
    axes[1].imshow(cv_overlay)
    axes[1].set_title(f"OpenCV SIFT\n{len(cv_kps)} keypoints", fontsize=13)
    axes[1].axis("off")
    fig.suptitle("Keypoint Comparison: This Implementation vs OpenCV", fontsize=15)
    plt.tight_layout()
    save_fig(fig, "9_comparison_keypoints.png")

    if this_descriptors and cv_descs is not None and len(cv_descs) > 0:
        this_mat = np.array([d["descriptor"] for d in this_descriptors], dtype=np.float32)
        # OpenCV stores descriptors as uint8 scaled x512; normalise to unit vectors
        cv_mat = cv_descs.astype(np.float32)
        cv_norms = np.linalg.norm(cv_mat, axis=1, keepdims=True)
        cv_mat = np.where(cv_norms > 0, cv_mat / cv_norms, cv_mat)

        # Nearest-neighbour distance: for each of these descriptors find the closest OpenCV one
        distances = np.array([
            np.sqrt(((cv_mat - d) ** 2).sum(axis=1)).min()
            for d in this_mat
        ])
        print(f"\n  Descriptor NN distance (this -> OpenCV):")
        print(f"    Mean:   {distances.mean():.4f}")
        print(f"    Median: {np.median(distances):.4f}")
        print(f"    Std:    {distances.std():.4f}")
        print("    (0 = identical, ~1.41 = orthogonal)")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(distances, bins=40, color="steelblue", edgecolor="white")
        ax.axvline(distances.mean(), color="red", linestyle="--",
                   label=f"Mean = {distances.mean():.3f}")
        ax.axvline(np.median(distances), color="orange", linestyle="--",
                   label=f"Median = {np.median(distances):.3f}")
        ax.set_xlabel("L2 Distance to Nearest OpenCV Descriptor")
        ax.set_ylabel("Count")
        ax.set_title("Descriptor Similarity: This SIFT -> Nearest OpenCV SIFT Descriptor")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "10_descriptor_distances.png")


#  Entry point 

def main():
    # Parse args
    parser = argparse.ArgumentParser(
        description="SIFT implementation with full octave pyramid")
    parser.add_argument("image", nargs="?", default="Assignment_MV_1_image.png",
                        help="Path to input image")
    parser.add_argument("--threshold", type=float, default=3.5,
                        help="DoG contrast threshold in raw [0,255] pixel units "
                             "(default: 3.5 ≈ OpenCV contrastThreshold=0.04/s*255)")
    parser.add_argument("--edge-threshold", type=float, default=10.0,
                        help="Edge rejection threshold r (default: 10.0)")
    parser.add_argument("--octaves", type=int, default=None,
                        help="Number of octaves (default: auto from image size)")
    parser.add_argument("--comparison", type=str, default=None, 
                        help="Choose to compare against OpenCV SIFT")
    args = parser.parse_args()

    image, gray = load_image(args.image)

    # Auto-compute octave count: enough levels until the image is ~32px on the short side
    n_octaves = args.octaves or max(1, int(np.log2(min(gray.shape))) - 2)
    print(f"Pyramid: {n_octaves} octaves, {N_SCALES_PER_OCTAVE} scales/octave, "
          f"sigma₀={SIGMA_0}")

    # Task 1 - build the octave pyramid
    print("\nBuilding octave pyramid...")
    octaves, sigmas_per_octave = build_octave_pyramid(gray, n_octaves)
    dog_octaves = compute_dog_pyramid(octaves)
    visualise_pyramid(octaves, dog_octaves)

    # Task 2 - detect, refine, and filter keypoints
    print("\nDetecting keypoints...")
    keypoints = detect_keypoints_pyramid(dog_octaves, sigmas_per_octave, args.threshold)
    # Refinement uses a lower contrast threshold (Lowe: 0.5 * contrastThreshold / s)
    # to avoid discarding borderline-valid keypoints that passed initial detection
    refine_threshold = args.threshold * 0.5
    keypoints = refine_keypoints(keypoints, dog_octaves, sigmas_per_octave, refine_threshold)
    keypoints = reject_edge_keypoints(keypoints, dog_octaves, args.edge_threshold)
    visualise_keypoints(image, keypoints, "5_keypoints_overlay.png",
                        f"Detected Keypoints - {len(keypoints)} after refinement and edge rejection")

    # Task 3 - gradients and orientation assignment
    print("\nComputing gradients and assigning orientations...")
    gx_octaves, gy_octaves = compute_gradients_pyramid(octaves)
    oriented_kps = assign_orientations(keypoints, gx_octaves, gy_octaves, sigmas_per_octave)
    visualise_oriented(image, oriented_kps, "8_oriented_keypoints.png",
                       "Oriented Keypoints (radius = sigma, line = dominant orientation)")

    # Task 4 - build 128-D descriptors with trilinear interpolation
    print(f"\nBuilding descriptors for {len(oriented_kps)} keypoints...")
    descriptors = build_descriptors(oriented_kps, gx_octaves, gy_octaves)
    print(f"Built {len(descriptors)} SIFT descriptors (128-D each)")

    # Benchmark against OpenCV
    if args.comparison is not None and args.comparison.lower() == "opencv":
        print("\nRunning OpenCV SIFT for comparison...")
        # Convert the raw [0,255] threshold to OpenCV's normalised [0,1] per-scale value:
        # OpenCV applies contrastThreshold / nOctaveLayers internally, so this un-does that
        # division to get the equivalent contrastThreshold argument.
        cv_contrast = (args.threshold / 255.0) * N_SCALES_PER_OCTAVE
        compare_with_opencv(image, gray, oriented_kps, descriptors,
                            contrast_threshold=cv_contrast,
                            edge_threshold=int(args.edge_threshold))

    print(f"\nAll figures saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
