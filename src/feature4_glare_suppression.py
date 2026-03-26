"""
Feature 4 — Reflection and Glare Suppression
Objective 2: Suppress water reflections and lighting artefacts in wet road images.

Wet roads produce bright reflection hotspots that create false high-gradient regions.
Without suppression, Feature 5 would mistake those bright patches for crack edges.

Steps:
  4a. Scharr gradient (X + Y)    — compute gradient magnitude across the whole image
  4b. Glare mask (brightness)    — flag pixels brighter than 230 as glare
  4c. Dilate glare mask          — expand mask to catch the faint halo around each hotspot
  4d. Apply road mask            — zero out gradient values outside the road area
  4e. Zero out glare regions     — set gradient to 0 wherever glare mask is active
      Compute glare_score        — glare pixels / road pixels (feeds RHI at 10% weight)

Requires Feature 1 + Feature 3 outputs:
  enhanced  — grayscale processed image (H x W, uint8) from F1
  road_mask — binary road area mask    (H x W, uint8) from F3

Run:  python feature4_glare_suppression.py path/to/image.jpg
      python feature4_glare_suppression.py              <- batch over data/raw/
Output: data/results/feature4/
"""

import cv2
import numpy as np
import os
import sys
import glob

# ── Folders ────────────────────────────────────────────────────────────────────
RAW_DIR     = "data/raw"
RESULTS_DIR = "data/results/feature4"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Thresholds ─────────────────────────────────────────────────────────────────
GLARE_THRESHOLD  = 230   # pixels brighter than this are flagged as glare (0–255)
GLARE_DILATE_K   = 5     # dilation kernel size — captures halo around bright spots

# ── Label drawing ──────────────────────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.52
FONT_COLOR = (255, 255, 0)   # yellow
FONT_THICK = 2


def put_label(img, text, pos=(10, 26)):
    """Yellow text with black background — readable on any image."""
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    x, y = pos
    cv2.rectangle(img, (x - 3, y - th - 5), (x + tw + 3, y + 3), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, FONT_COLOR, FONT_THICK)
    return img


def to_display(arr, label):
    """Normalise any single-channel array to 0–255 BGR and add a label."""
    if arr.dtype != np.uint8:
        # Normalise float/64-bit gradient maps to visible range
        norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
        arr  = norm.astype(np.uint8)
    if len(arr.shape) == 2:
        d = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        d = arr.copy()
    put_label(d, label)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4a — Scharr gradient magnitude
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gradient(enhanced):
    """
    Scharr operator computes the gradient (rate of brightness change) in X and Y.
    Scharr is more accurate than Sobel at detecting diagonal edges — important
    because cracks run in all directions.
    High gradient magnitude = strong edge (could be a real crack OR glare artefact).
    """
    sx        = cv2.Scharr(enhanced, cv2.CV_64F, 1, 0)   # horizontal gradient
    sy        = cv2.Scharr(enhanced, cv2.CV_64F, 0, 1)   # vertical gradient
    magnitude = np.sqrt(sx ** 2 + sy ** 2)               # combine into one map

    mean_grad = float(np.mean(magnitude))
    max_grad  = float(np.max(magnitude))
    label     = f"4a Scharr gradient | mean={mean_grad:.1f}  max={max_grad:.1f}"
    print(f"  [4a] Scharr gradient done — mean={mean_grad:.1f}  max={max_grad:.1f}")
    return magnitude, to_display(magnitude, label)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4b — Brightness threshold → glare mask
# ═══════════════════════════════════════════════════════════════════════════════

def create_glare_mask(enhanced):
    """
    Any pixel brighter than GLARE_THRESHOLD (default 230) is flagged as glare.
    These abnormally bright regions are water reflection hotspots.
    Output: binary mask, 255 = glare pixel, 0 = normal pixel.
    """
    _, glare_mask = cv2.threshold(enhanced, GLARE_THRESHOLD, 255, cv2.THRESH_BINARY)
    glare_px  = cv2.countNonZero(glare_mask)
    glare_pct = 100.0 * glare_px / glare_mask.size
    label     = f"4b Glare mask | thresh={GLARE_THRESHOLD} | {glare_pct:.1f}% bright px"
    print(f"  [4b] Glare mask — {glare_px} px ({glare_pct:.1f}%) flagged as glare")
    return glare_mask, to_display(glare_mask, label), glare_px


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4c — Dilate glare mask (catch the halo around each hotspot)
# ═══════════════════════════════════════════════════════════════════════════════

def dilate_glare_mask(glare_mask):
    """
    The pixels just outside each bright hotspot also produce false gradient spikes
    (the 'halo' effect). Dilation expands the mask slightly to cover those halos.
    A 5x5 kernel with 1 iteration gives a small, controlled expansion.
    """
    kernel       = np.ones((GLARE_DILATE_K, GLARE_DILATE_K), np.uint8)
    dilated      = cv2.dilate(glare_mask, kernel, iterations=1)
    dilated_px   = cv2.countNonZero(dilated)
    extra_px     = dilated_px - cv2.countNonZero(glare_mask)
    label        = f"4c Dilated glare | k={GLARE_DILATE_K}x{GLARE_DILATE_K} | +{extra_px} halo px"
    print(f"  [4c] Dilation done — halo expanded by {extra_px} px (total glare: {dilated_px} px)")
    return dilated, to_display(dilated, label)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4d — Apply road mask (restrict everything to road pixels)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_road_mask(magnitude, road_mask):
    """
    Zero out gradient values outside the road area.
    We only care about glare and cracks ON the road — not on buildings or sky.
    """
    mag_u8    = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    masked    = cv2.bitwise_and(mag_u8, mag_u8, mask=road_mask)
    road_px   = cv2.countNonZero(road_mask)
    label     = f"4d Road-masked gradient | road_px={road_px}"
    print(f"  [4d] Road mask applied — {road_px} road pixels retained")
    return masked, to_display(masked, label), road_px


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4e — Zero out glare regions + compute glare_score
# ═══════════════════════════════════════════════════════════════════════════════

def suppress_glare(masked_grad, dilated_glare, road_mask, road_px):
    """
    Sets gradient to 0 wherever the dilated glare mask is active.
    Real crack gradients remain; reflection-induced gradients are removed.

    glare_score = glare pixels inside road / total road pixels
    Range: 0.0 (no glare) → 1.0 (entire road surface is glare).
    Feeds into the RHI formula (Feature 12) at 10% weight.
    """
    suppressed = masked_grad.copy()
    suppressed[dilated_glare == 255] = 0    # zero out glare regions

    # Count glare pixels that overlap with the road area
    glare_on_road   = cv2.bitwise_and(dilated_glare, dilated_glare, mask=road_mask)
    glare_road_px   = cv2.countNonZero(glare_on_road)
    glare_score     = glare_road_px / road_px if road_px > 0 else 0.0

    label = f"4e Suppressed | glare_score={glare_score:.4f} ({100*glare_score:.1f}% of road)"
    print(f"  [4e] Glare suppressed — glare_score = {glare_score:.4f}  ({100*glare_score:.1f}% of road)")
    return suppressed, to_display(suppressed, label), glare_score


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def suppress_glare_full(enhanced, road_mask, base_name="image"):
    """
    Run all five glare-suppression steps on one image.
    Saves each step output + a 2x3 comparison grid.
    Returns suppressed gradient map and glare_score for use by the pipeline.

    Parameters
    ----------
    enhanced  : grayscale numpy array (480 x 640, uint8) — from F1
    road_mask : binary numpy array    (480 x 640, uint8) — from F3
    base_name : str — image file stem used for output folder naming

    Returns
    -------
    suppressed  : uint8 numpy array (480 x 640) — glare-free gradient map
    glare_score : float 0.0–1.0 — fraction of road covered by glare
    """
    out_dir = os.path.join(RESULTS_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n── Feature 4: {base_name} ──")

    # ── Input display panels ──────────────────────────────────────────────────
    enh_disp  = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    put_label(enh_disp, "F1 Enhanced (gray) — input")

    mask_disp = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
    put_label(mask_disp, "F3 Road mask — input")

    # ── Run steps ─────────────────────────────────────────────────────────────
    magnitude,    disp_grad              = compute_gradient(enhanced)
    glare_mask,   disp_glare, glare_px  = create_glare_mask(enhanced)
    dilated,      disp_dilated           = dilate_glare_mask(glare_mask)
    masked_grad,  disp_masked, road_px  = apply_road_mask(magnitude, road_mask)
    suppressed,   disp_supp, glare_score = suppress_glare(masked_grad, dilated, road_mask, road_px)

    # ── Difference map: what was removed by suppression ───────────────────────
    diff        = cv2.absdiff(masked_grad, suppressed)
    diff_bright = cv2.convertScaleAbs(diff, alpha=4)   # amplify so changes are visible
    diff_disp   = cv2.cvtColor(diff_bright, cv2.COLOR_GRAY2BGR)
    removed_px  = int(np.sum(diff > 5))
    put_label(diff_disp, f"Removed gradient (x4) | {removed_px} px suppressed")

    # ── Overlay: glare regions shown in red on the original ───────────────────
    overlay = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    red_layer = overlay.copy()
    red_layer[dilated == 255] = (0, 0, 220)
    blended = cv2.addWeighted(overlay, 0.6, red_layer, 0.4, 0)
    put_label(blended, f"Glare overlay (red) | score={glare_score:.4f}")
    put_label(blended, f"glare={100*glare_score:.1f}% of road | thresh={GLARE_THRESHOLD}", pos=(10, 50))

    # ── Save individual step images ───────────────────────────────────────────
    cv2.imwrite(os.path.join(out_dir, "step4a_gradient.jpg"),  disp_grad)
    cv2.imwrite(os.path.join(out_dir, "step4b_glare.jpg"),     disp_glare)
    cv2.imwrite(os.path.join(out_dir, "step4c_dilated.jpg"),   disp_dilated)
    cv2.imwrite(os.path.join(out_dir, "step4d_masked.jpg"),    disp_masked)
    cv2.imwrite(os.path.join(out_dir, "step4e_suppressed.jpg"), disp_supp)
    cv2.imwrite(os.path.join(out_dir, "step4e_overlay.jpg"),   blended)
    cv2.imwrite(os.path.join(out_dir, "step4e_diff.jpg"),      diff_disp)

    # ── Build 2x3 comparison grid ─────────────────────────────────────────────
    row1 = np.hstack([enh_disp,    disp_grad,    disp_glare])
    row2 = np.hstack([disp_dilated, disp_supp,   blended])
    grid = np.vstack([row1, row2])
    scale = min(1.0, 1280 / grid.shape[1])
    grid  = cv2.resize(grid, (0, 0), fx=scale, fy=scale)

    grid_path = os.path.join(out_dir, "FEATURE4_grid.jpg")
    cv2.imwrite(grid_path, grid)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"  Glare pixels    : {glare_px} (before dilation)")
    print(f"  Road pixels     : {road_px}")
    print(f"  Glare score     : {glare_score:.4f}  ({100*glare_score:.1f}% of road)")
    print(f"  Gradient px rm  : {removed_px}")
    print(f"  Saved to        : {out_dir}/")

    # ── Show grid window ──────────────────────────────────────────────────────
    cv2.imshow(f"Feature 4 — {base_name}", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return suppressed, glare_score   # pass suppressed into F5; glare_score into F12


# ═══════════════════════════════════════════════════════════════════════════════
# Inline Features 1 + 3 — used only when running F4 standalone
# ═══════════════════════════════════════════════════════════════════════════════

def _load_enhance_roi(image_path):
    """Minimal F1 + F3 replication so F4 can run standalone."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return None, None

    img   = cv2.resize(img, (640, 480))
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh   = clahe.apply(gray)
    table = np.array([(i / 255.0) ** (1 / 1.3) * 255 for i in range(256)], dtype=np.uint8)
    enh   = cv2.LUT(enh, table)
    enh   = cv2.medianBlur(enh, 5)

    # Minimal ROI extraction (F3)
    edges   = cv2.Canny(enh, 50, 150)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    road_mask = np.zeros_like(enh)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= 5000:
            cv2.drawContours(road_mask, [largest], -1, 255, thickness=cv2.FILLED)
        else:
            road_mask[:] = 255   # fallback
    else:
        road_mask[:] = 255       # fallback

    return enh, road_mask


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point — single image or batch over data/raw/
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_paths = [sys.argv[1]]
    else:
        patterns = [
            os.path.join(RAW_DIR, "**", "*.jpg"),
            os.path.join(RAW_DIR, "**", "*.png"),
            os.path.join(RAW_DIR, "**", "*.jpeg"),
        ]
        image_paths = []
        for p in patterns:
            image_paths.extend(glob.glob(p, recursive=True))

    if not image_paths:
        print(f"[ERROR] No images found under '{RAW_DIR}/'")
        print("  Single image : python feature4_glare_suppression.py path/to/img.jpg")
        print("  Batch        : put images in data/raw/ subfolders")
        sys.exit(1)

    print(f"[INFO] Found {len(image_paths)} image(s) — processing all...\n")
    for img_path in image_paths:
        enh, road_mask = _load_enhance_roi(img_path)
        if enh is None:
            continue
        name = os.path.splitext(os.path.basename(img_path))[0]
        suppress_glare_full(enh, road_mask, base_name=name)

    print("\n[DONE] Feature 4 complete.")