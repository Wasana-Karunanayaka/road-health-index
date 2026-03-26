"""
Feature 6 — Pothole Detection
Objective 4: Detect and segment large dark collapsed areas (potholes).

Potholes are wide, dark, roughly circular/irregular regions — visually different
from cracks (thin lines). This feature uses Otsu global thresholding + elliptical
morphology + connected component analysis to find and measure each pothole.

Steps:
  6a. Otsu global threshold      — isolates all dark pixels (potholes are dark)
  6b. Apply road mask            — removes detections outside the road area
  6c. Morphological opening      — removes thin lines and small noise (7x7 ellipse)
  6d. Morphological closing      — fills gaps inside pothole blobs (7x7 ellipse)
  6e. Connected component filter — labels each blob; keeps only area >= 500 px
      Compute pothole_score      — total pothole pixels / road pixels (feeds F12)

Requires Feature 1 + Feature 3 outputs:
  enhanced  — grayscale processed image (H x W, uint8) from F1
  road_mask — binary road area mask    (H x W, uint8) from F3

Run:  python feature6_pothole_detection.py path/to/image.jpg
      python feature6_pothole_detection.py              <- batch over data/raw/
Output: data/results/feature6/
"""

import cv2
import numpy as np
import os
import sys
import glob

# ── Folders ────────────────────────────────────────────────────────────────────
RAW_DIR     = "data/raw"
RESULTS_DIR = "data/results/feature6"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Morphological kernel size ──────────────────────────────────────────────────
ELLIPSE_K = 7      # elliptical kernel — better suited to circular pothole shapes

# ── Minimum pothole area (pixels) ─────────────────────────────────────────────
MIN_POTHOLE_AREA = 500   # blobs smaller than this are noise, not potholes

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
# STEP 6a — Otsu global threshold
# ═══════════════════════════════════════════════════════════════════════════════

def otsu_threshold(enhanced):
    """
    Otsu automatically finds the best single threshold to split the histogram.
    Pixels darker than the threshold → white (255) in output.
    Potholes are much darker than surrounding road surface so they flip to white.
    """
    otsu_val, binary = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    white_px  = cv2.countNonZero(binary)
    white_pct = 100.0 * white_px / binary.size
    label     = f"6a Otsu thresh={int(otsu_val)} | white={white_pct:.1f}%"
    print(f"  [6a] Otsu threshold={int(otsu_val)} — {white_px} px ({white_pct:.1f}%) dark")
    return binary, to_display(binary, label), int(otsu_val)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6b — Apply road mask
# ═══════════════════════════════════════════════════════════════════════════════

def apply_road_mask(binary, road_mask):
    """
    Zeros out detections outside the road boundary.
    Dark shadows from buildings or trees outside the road would be false positives.
    """
    masked    = cv2.bitwise_and(binary, binary, mask=road_mask)
    road_px   = cv2.countNonZero(road_mask)
    dark_px   = cv2.countNonZero(masked)
    dark_pct  = 100.0 * dark_px / road_px if road_px > 0 else 0.0
    label     = f"6b Road-masked | road_px={road_px} | dark={dark_pct:.1f}% of road"
    print(f"  [6b] Road mask applied — {dark_px} dark px ({dark_pct:.1f}% of road)")
    return masked, to_display(masked, label), road_px


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6c — Morphological opening (remove thin lines and noise)
# ═══════════════════════════════════════════════════════════════════════════════

def morph_open(masked):
    """
    Erosion then dilation using an elliptical 7x7 kernel.
    Elliptical kernel is better at preserving round/circular shapes.
    Thin crack lines (from F5's territory) and small noise dots are erased.
    Only wide, compact dark regions (potholes) survive.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ELLIPSE_K, ELLIPSE_K))
    opened = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)
    removed = cv2.countNonZero(masked) - cv2.countNonZero(opened)
    label  = f"6c Opening ellipse {ELLIPSE_K}x{ELLIPSE_K} | thin lines removed={removed}px"
    print(f"  [6c] Opening done — {removed} thin/noise pixels removed")
    return opened, to_display(opened, label)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6d — Morphological closing (fill gaps inside pothole blobs)
# ═══════════════════════════════════════════════════════════════════════════════

def morph_close(opened):
    """
    Dilation then erosion using the same elliptical 7x7 kernel.
    Fills any holes or gaps inside detected pothole regions.
    Makes each pothole a solid filled shape — needed for accurate area measurement.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ELLIPSE_K, ELLIPSE_K))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    filled = cv2.countNonZero(closed) - cv2.countNonZero(opened)
    label  = f"6d Closing ellipse {ELLIPSE_K}x{ELLIPSE_K} | holes filled={filled}px"
    print(f"  [6d] Closing done — {filled} hole pixels filled")
    return closed, to_display(closed, label)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6e — Connected component analysis + area filter
# ═══════════════════════════════════════════════════════════════════════════════

def connected_components_filter(closed, road_px):
    """
    Labels each separate white blob with a unique ID.
    Measures area, bounding box, and centroid of every blob.
    Keeps only blobs with area >= MIN_POTHOLE_AREA — the rest are noise.
    Computes pothole_score = total_pothole_pixels / road_pixels for F12.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)

    pothole_mask  = np.zeros_like(closed)
    pothole_areas = []

    for i in range(1, n):   # label 0 is background — skip it
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= MIN_POTHOLE_AREA:
            pothole_mask[labels == i] = 255
            pothole_areas.append(area)

    pothole_px    = cv2.countNonZero(pothole_mask)
    pothole_score = pothole_px / road_px if road_px > 0 else 0.0
    count         = len(pothole_areas)

    label = (f"6e CCL filter >= {MIN_POTHOLE_AREA}px | "
             f"potholes={count} | score={pothole_score:.4f}")
    print(f"  [6e] {n-1} blobs found → {count} potholes kept "
          f"(areas: {pothole_areas})")
    print(f"  [6e] pothole_score = {pothole_score:.4f}  "
          f"({100*pothole_score:.1f}% of road)")

    return pothole_mask, pothole_areas, pothole_score, to_display(pothole_mask, label)


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_potholes(enhanced, road_mask, base_name="image"):
    """
    Run all five pothole-detection steps on one image.
    Saves each step output + a 2x3 comparison grid.
    Returns pothole_mask, pothole_areas, and pothole_score for the pipeline.

    Parameters
    ----------
    enhanced  : grayscale numpy array (480 x 640, uint8) — from F1
    road_mask : binary numpy array    (480 x 640, uint8) — from F3
    base_name : str — image file stem used for output folder naming

    Returns
    -------
    pothole_mask  : binary numpy array (480 x 640, uint8) — white = pothole pixel
    pothole_areas : list of ints — area in pixels of each detected pothole
    pothole_score : float 0.0–1.0 — pothole coverage fraction of road
    """
    out_dir = os.path.join(RESULTS_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n── Feature 6: {base_name} ──")

    # ── Input display panels ──────────────────────────────────────────────────
    enh_disp  = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    put_label(enh_disp, "F1 Enhanced (gray) — input")

    mask_disp = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
    put_label(mask_disp, "F3 Road mask — input")

    # ── Run steps ─────────────────────────────────────────────────────────────
    binary,       disp_binary, otsu_val          = otsu_threshold(enhanced)
    masked,       disp_masked, road_px           = apply_road_mask(binary, road_mask)
    opened,       disp_opened                    = morph_open(masked)
    closed,       disp_closed                    = morph_close(opened)
    pothole_mask, pothole_areas, pothole_score, disp_potholes = \
        connected_components_filter(closed, road_px)

    # ── Annotate the final pothole mask panel ─────────────────────────────────
    final_disp = disp_potholes.copy()
    put_label(final_disp,
              f"Pothole mask | count={len(pothole_areas)} | score={pothole_score:.4f}")
    put_label(final_disp,
              f"total_area={sum(pothole_areas)}px  road_px={road_px}", pos=(10, 50))

    # ── Colour overlay: potholes in red on the enhanced image ─────────────────
    overlay   = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    red_layer = overlay.copy()
    red_layer[pothole_mask == 255] = (0, 0, 220)   # red over pothole areas
    blended   = cv2.addWeighted(overlay, 0.6, red_layer, 0.4, 0)
    put_label(blended, f"Pothole overlay | score={pothole_score:.4f}")
    put_label(blended,
              f"count={len(pothole_areas)}  coverage={100*pothole_score:.1f}% of road",
              pos=(10, 50))

    # ── Draw bounding boxes + area labels on the overlay ──────────────────────
    if pothole_areas:
        _, _, stats_full, _ = cv2.connectedComponentsWithStats(pothole_mask)
        for i in range(1, len(stats_full)):
            x = int(stats_full[i, cv2.CC_STAT_LEFT])
            y = int(stats_full[i, cv2.CC_STAT_TOP])
            w = int(stats_full[i, cv2.CC_STAT_WIDTH])
            h = int(stats_full[i, cv2.CC_STAT_HEIGHT])
            a = int(stats_full[i, cv2.CC_STAT_AREA])
            cv2.rectangle(blended, (x, y), (x + w, y + h), (0, 255, 255), 1)  # cyan box
            cv2.putText(blended, f"{a}px", (x + 2, y + h - 4),
                        FONT, 0.38, (0, 255, 255), 1)

    # ── Save individual step images ───────────────────────────────────────────
    cv2.imwrite(os.path.join(out_dir, "step6a_otsu.jpg"),      disp_binary)
    cv2.imwrite(os.path.join(out_dir, "step6b_masked.jpg"),    disp_masked)
    cv2.imwrite(os.path.join(out_dir, "step6c_opened.jpg"),    disp_opened)
    cv2.imwrite(os.path.join(out_dir, "step6d_closed.jpg"),    disp_closed)
    cv2.imwrite(os.path.join(out_dir, "step6e_potholes.jpg"),  final_disp)
    cv2.imwrite(os.path.join(out_dir, "step6e_overlay.jpg"),   blended)

    # ── Build 2x3 comparison grid ─────────────────────────────────────────────
    row1 = np.hstack([enh_disp,   disp_binary, disp_masked])
    row2 = np.hstack([disp_opened, final_disp,  blended])
    grid = np.vstack([row1, row2])
    scale = min(1.0, 1280 / grid.shape[1])
    grid  = cv2.resize(grid, (0, 0), fx=scale, fy=scale)

    grid_path = os.path.join(out_dir, "FEATURE6_grid.jpg")
    cv2.imwrite(grid_path, grid)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"  Potholes found  : {len(pothole_areas)}")
    if pothole_areas:
        print(f"  Areas (px)      : {pothole_areas}")
        print(f"  Largest pothole : {max(pothole_areas)} px")
        print(f"  Total area      : {sum(pothole_areas)} px")
    print(f"  Road pixels     : {road_px}")
    print(f"  Pothole score   : {pothole_score:.4f}  ({100*pothole_score:.1f}% of road)")
    print(f"  Saved to        : {out_dir}/")

    # ── Show grid window ──────────────────────────────────────────────────────
    cv2.imshow(f"Feature 6 — {base_name}", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pothole_mask, pothole_areas, pothole_score
    # pothole_mask  -> F7 (geometry classification)
    # pothole_score -> F12 (20% weight in RHI formula)


# ═══════════════════════════════════════════════════════════════════════════════
# Inline Features 1 + 3 — used only when running F6 standalone
# ═══════════════════════════════════════════════════════════════════════════════

def _load_enhance_roi(image_path):
    """Minimal F1 + F3 replication so F6 can run standalone."""
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
        print("  Single image : python feature6_pothole_detection.py path/to/img.jpg")
        print("  Batch        : put images in data/raw/ subfolders")
        sys.exit(1)

    print(f"[INFO] Found {len(image_paths)} image(s) — processing all...\n")
    for img_path in image_paths:
        enh, road_mask = _load_enhance_roi(img_path)
        if enh is None:
            continue
        name = os.path.splitext(os.path.basename(img_path))[0]
        detect_potholes(enh, road_mask, base_name=name)

    print("\n[DONE] Feature 6 complete.")