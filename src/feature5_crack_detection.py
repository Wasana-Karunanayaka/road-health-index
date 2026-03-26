"""
Feature 5 — Crack Detection and Segmentation
Objective 4: Detect and segment road cracks.

Produces a binary crack map (white = crack) and a skeleton (single-pixel
centrelines). Every step saves a labelled image so you can verify visually.

Steps:
  5a. Adaptive threshold       — dark cracks become white; handles uneven lighting
  5b. Apply road mask          — zero out any detections outside the road area
  5c. Morphological opening    — removes small noise dots (3x3 kernel)
  5d. Morphological closing    — bridges small gaps between crack fragments (5x5)
  5e. Skeletonisation          — thins cracks to single-pixel centrelines

Requires Feature 1 + Feature 3 outputs:
  enhanced  — grayscale processed image (H x W, uint8) from F1
  road_mask — binary road area mask    (H x W, uint8) from F3

Run:  python feature5_crack_detection.py path/to/image.jpg
      python feature5_crack_detection.py              <- batch over data/raw/
Output: data/results/feature5/
"""

import cv2
import numpy as np
import os
import sys
import glob
from skimage.morphology import skeletonize

# ── Folders ────────────────────────────────────────────────────────────────────
RAW_DIR     = "data/raw"
RESULTS_DIR = "data/results/feature5"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Adaptive threshold parameters ─────────────────────────────────────────────
ADAPTIVE_BLOCK  = 11   # neighbourhood size for local threshold (must be odd)
ADAPTIVE_C      = 2    # constant subtracted from local mean

# ── Morphological kernel sizes ─────────────────────────────────────────────────
OPEN_K  = 3    # opening: removes noise dots smaller than this
CLOSE_K = 5    # closing: fills gaps smaller than this

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
    """Convert single-channel image to BGR and add a label."""
    if len(arr.shape) == 2:
        d = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        d = arr.copy()
    put_label(d, label)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5a — Adaptive threshold
# ═══════════════════════════════════════════════════════════════════════════════

def adaptive_threshold(enhanced):
    """
    Computes a separate threshold for every 11x11 neighbourhood.
    Dark crack pixels fall below the local mean → they become white (255).
    Works on images with uneven lighting that would fool a global threshold.
    """
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK, ADAPTIVE_C
    )
    white_px  = cv2.countNonZero(binary)
    white_pct = 100.0 * white_px / binary.size
    label     = f"5a Adaptive thresh | block={ADAPTIVE_BLOCK} C={ADAPTIVE_C} | white={white_pct:.1f}%"
    print(f"  [5a] Adaptive threshold done — {white_px} px ({white_pct:.1f}%) marked")
    return binary, to_display(binary, label)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5b — Apply road mask
# ═══════════════════════════════════════════════════════════════════════════════

def apply_road_mask(binary, road_mask):
    """
    Zeros out detections outside the road boundary.
    Without this, cracks on walls, footpaths, and grass would be counted.
    """
    masked    = cv2.bitwise_and(binary, binary, mask=road_mask)
    road_px   = cv2.countNonZero(road_mask)
    crack_px  = cv2.countNonZero(masked)
    crack_pct = 100.0 * crack_px / road_px if road_px > 0 else 0.0
    label     = f"5b Road-masked | road_px={road_px} | crack={crack_pct:.1f}% of road"
    print(f"  [5b] Road mask applied — {crack_px} crack px ({crack_pct:.1f}% of road)")
    return masked, to_display(masked, label), road_px


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5c — Morphological opening (noise removal)
# ═══════════════════════════════════════════════════════════════════════════════

def morph_open(masked):
    """
    Erosion then dilation.
    Any white region smaller than the 3x3 kernel is erased — these are noise
    dots, not real cracks. Crack structures (larger) survive intact.
    """
    k3     = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_K, OPEN_K))
    opened = cv2.morphologyEx(masked, cv2.MORPH_OPEN, k3)
    removed = cv2.countNonZero(masked) - cv2.countNonZero(opened)
    label  = f"5c Opening k={OPEN_K}x{OPEN_K} | noise dots removed={removed}px"
    print(f"  [5c] Opening done — {removed} noise pixels removed")
    return opened, to_display(opened, label)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5d — Morphological closing (gap filling)
# ═══════════════════════════════════════════════════════════════════════════════

def morph_close(opened):
    """
    Dilation then erosion.
    Bridges small gaps (< 5 px) between nearby crack fragments.
    A crack that appears as two separate segments becomes one continuous line.
    """
    k5     = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_K, CLOSE_K))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k5)
    added  = cv2.countNonZero(closed) - cv2.countNonZero(opened)
    label  = f"5d Closing k={CLOSE_K}x{CLOSE_K} | gap px filled={added}"
    print(f"  [5d] Closing done — {added} gap pixels filled")
    return closed, to_display(closed, label)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5e — Skeletonisation
# ═══════════════════════════════════════════════════════════════════════════════

def skeletonise(closed):
    """
    Thins every crack region to a single-pixel-wide centreline.
    Makes crack length measurement and orientation analysis accurate in F8.
    skimage expects a 0/1 binary array — divide by 255 first.
    """
    skel       = skeletonize(closed // 255).astype(np.uint8) * 255
    skel_px    = cv2.countNonZero(skel)
    label      = f"5e Skeleton | centreline_px={skel_px}"
    print(f"  [5e] Skeletonisation done — {skel_px} centreline pixels")
    return skel, to_display(skel, label)


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_cracks(enhanced, road_mask, base_name="image"):
    """
    Run all five crack-detection steps on one image.
    Saves each step output + a 2x3 comparison grid.
    Returns crack_map and skeleton for use by the pipeline.

    Parameters
    ----------
    enhanced  : grayscale numpy array (480 x 640, uint8) — from F1
    road_mask : binary numpy array    (480 x 640, uint8) — from F3
    base_name : str — image file stem used for output folder naming

    Returns
    -------
    crack_map : binary numpy array (480 x 640, uint8) — white = crack pixel
    skeleton  : binary numpy array (480 x 640, uint8) — single-pixel centrelines
    """
    out_dir = os.path.join(RESULTS_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n── Feature 5: {base_name} ──")

    # ── Input display panels ──────────────────────────────────────────────────
    enh_disp  = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    put_label(enh_disp, "F1 Enhanced (gray) — input")

    mask_disp = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
    put_label(mask_disp, "F3 Road mask — input")

    # ── Run steps ─────────────────────────────────────────────────────────────
    binary,   disp_binary                    = adaptive_threshold(enhanced)
    masked,   disp_masked,   road_px         = apply_road_mask(binary, road_mask)
    opened,   disp_opened                    = morph_open(masked)
    crack_map, disp_closed                   = morph_close(opened)
    skeleton, disp_skel                      = skeletonise(crack_map)

    # ── Compute crack density ─────────────────────────────────────────────────
    crack_px      = cv2.countNonZero(crack_map)
    crack_density = crack_px / road_px if road_px > 0 else 0.0

    # ── Annotate the final crack map panel ────────────────────────────────────
    final_disp = disp_closed.copy()
    put_label(final_disp,
              f"Crack map | density={crack_density:.4f} ({100*crack_density:.1f}% of road)")
    put_label(final_disp,
              f"crack_px={crack_px}  road_px={road_px}", pos=(10, 50))

    # ── Colour overlay: cracks in cyan on the enhanced image ──────────────────
    overlay    = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    cyan_layer = overlay.copy()
    cyan_layer[crack_map == 255] = (220, 220, 0)   # cyan over crack areas
    blended    = cv2.addWeighted(overlay, 0.6, cyan_layer, 0.4, 0)
    put_label(blended, f"Crack overlay | density={crack_density:.4f}")
    put_label(blended, f"crack={100*crack_density:.1f}% of road", pos=(10, 50))

    # ── Skeleton overlay: skeleton in green on enhanced image ─────────────────
    skel_overlay = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    skel_overlay[skeleton == 255] = (0, 255, 0)   # green centrelines
    put_label(skel_overlay, f"Skeleton overlay | centreline_px={cv2.countNonZero(skeleton)}")

    # ── Save individual step images ───────────────────────────────────────────
    cv2.imwrite(os.path.join(out_dir, "step5a_adaptive.jpg"),   disp_binary)
    cv2.imwrite(os.path.join(out_dir, "step5b_masked.jpg"),     disp_masked)
    cv2.imwrite(os.path.join(out_dir, "step5c_opened.jpg"),     disp_opened)
    cv2.imwrite(os.path.join(out_dir, "step5d_crackmap.jpg"),   final_disp)
    cv2.imwrite(os.path.join(out_dir, "step5e_skeleton.jpg"),   disp_skel)
    cv2.imwrite(os.path.join(out_dir, "step5e_overlay.jpg"),    blended)
    cv2.imwrite(os.path.join(out_dir, "step5e_skel_overlay.jpg"), skel_overlay)

    # ── Build 2x3 comparison grid ─────────────────────────────────────────────
    row1 = np.hstack([enh_disp,    disp_binary,  disp_masked])
    row2 = np.hstack([disp_opened, final_disp,   blended])
    grid = np.vstack([row1, row2])
    scale = min(1.0, 1280 / grid.shape[1])
    grid  = cv2.resize(grid, (0, 0), fx=scale, fy=scale)

    grid_path = os.path.join(out_dir, "FEATURE5_grid.jpg")
    cv2.imwrite(grid_path, grid)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"  Crack pixels    : {crack_px}")
    print(f"  Road pixels     : {road_px}")
    print(f"  Crack density   : {crack_density:.4f}  ({100*crack_density:.1f}% of road)")
    print(f"  Skeleton px     : {cv2.countNonZero(skeleton)}")
    print(f"  Saved to        : {out_dir}/")

    # ── Show grid window ──────────────────────────────────────────────────────
    cv2.imshow(f"Feature 5 — {base_name}", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return crack_map, skeleton   # crack_map -> F7, F8, F12 | skeleton -> F8


# ═══════════════════════════════════════════════════════════════════════════════
# Inline Features 1 + 3 — used only when running F5 standalone
# ═══════════════════════════════════════════════════════════════════════════════

def _load_enhance_roi(image_path):
    """Minimal F1 + F3 replication so F5 can run standalone."""
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
    edges  = cv2.Canny(enh, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    road_mask = np.zeros_like(enh)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= 5000:
            cv2.drawContours(road_mask, [largest], -1, 255, thickness=cv2.FILLED)
        else:
            road_mask[:] = 255   # fallback: use full image
    else:
        road_mask[:] = 255       # fallback: use full image

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
        print("  Single image : python feature5_crack_detection.py path/to/img.jpg")
        print("  Batch        : put images in data/raw/ subfolders")
        sys.exit(1)

    print(f"[INFO] Found {len(image_paths)} image(s) — processing all...\n")
    for img_path in image_paths:
        enh, road_mask = _load_enhance_roi(img_path)
        if enh is None:
            continue
        name = os.path.splitext(os.path.basename(img_path))[0]
        detect_cracks(enh, road_mask, base_name=name)

    print("\n[DONE] Feature 5 complete.")