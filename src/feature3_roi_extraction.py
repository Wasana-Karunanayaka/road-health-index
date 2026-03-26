"""
Feature 3 — Road Area Extraction (ROI Detection)
Objective 3: Extract the road surface region from surrounding background areas.

Creates a binary mask showing exactly which pixels are road surface.
All features after this (F4, F5, F6) run ONLY on road pixels.

Steps:
  3a. Canny edge detection       — finds strong boundaries in the image
  3b. Morphological closing      — connects broken edge segments into a closed boundary
  3c. Largest contour selection  — assumes the biggest closed region is the road
  3d. Fill the contour           — solid white road mask on black background
  3e. Fallback check             — if nothing found, use full image as ROI

Requires Feature 1 output:
  enhanced — grayscale processed image (H x W, uint8)

Run:  python feature3_roi_extraction.py path/to/image.jpg
      python feature3_roi_extraction.py              <- batch over data/raw/
Output: data/results/feature3/
"""

import cv2
import numpy as np
import os
import sys
import glob

# ── Folders ────────────────────────────────────────────────────────────────────
RAW_DIR     = "data/raw"
RESULTS_DIR = "data/results/feature3"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Canny thresholds ───────────────────────────────────────────────────────────
CANNY_LOW      = 50    # weak edges below this are discarded
CANNY_HIGH     = 150   # strong edges above this are always kept

# ── Morphological closing kernel size ─────────────────────────────────────────
CLOSE_KERNEL   = 15    # large kernel fills bigger gaps in road boundary lines

# ── Minimum road area (pixels) — anything smaller is not a road ───────────────
MIN_ROAD_AREA  = 5000  # if no contour is this big, fall back to full image

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


def to_display(gray_or_mask, label):
    """Convert single-channel image to BGR and add a label."""
    if len(gray_or_mask.shape) == 2:
        d = cv2.cvtColor(gray_or_mask, cv2.COLOR_GRAY2BGR)
    else:
        d = gray_or_mask.copy()
    put_label(d, label)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3a — Canny edge detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_edges(enhanced):
    """
    Finds strong edges in the enhanced grayscale image.
    The road boundary shows up as a strong edge (big brightness change).
    Output: binary image where 255 = edge pixel, 0 = no edge.
    """
    edges     = cv2.Canny(enhanced, CANNY_LOW, CANNY_HIGH)
    edge_px   = cv2.countNonZero(edges)
    label     = f"3a Canny edges | thresh={CANNY_LOW}/{CANNY_HIGH} | edge_px={edge_px}"
    print(f"  [3a] Canny done — {edge_px} edge pixels found")
    return edges, to_display(edges, label)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3b — Morphological closing (connect broken boundary lines)
# ═══════════════════════════════════════════════════════════════════════════════

def close_edges(edges):
    """
    Canny often leaves small gaps in the road boundary outline.
    Closing (dilate then erode) bridges those gaps.
    A large 15x15 kernel is used so even bigger gaps get filled.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_KERNEL, CLOSE_KERNEL))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    closed_px = cv2.countNonZero(closed)
    label  = f"3b Closing k={CLOSE_KERNEL}x{CLOSE_KERNEL} | filled_px={closed_px}"
    print(f"  [3b] Closing done — {closed_px} px after gap filling")
    return closed, to_display(closed, label)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3c — Find the largest contour
# ═══════════════════════════════════════════════════════════════════════════════

def find_largest_contour(closed):
    """
    Finds all closed outlines in the edge image.
    The largest one by area is assumed to be the road boundary.
    Returns None if no contour meets the minimum area threshold.
    """
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("  [3c] No contours found at all — will use fallback")
        return None, 0

    largest  = max(contours, key=cv2.contourArea)
    area     = int(cv2.contourArea(largest))
    n        = len(contours)
    print(f"  [3c] {n} contours found — largest area = {area} px")

    if area < MIN_ROAD_AREA:
        print(f"  [3c] Largest area {area} < MIN_ROAD_AREA {MIN_ROAD_AREA} — will use fallback")
        return None, area

    return largest, area


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3d — Fill the contour to create a solid road mask
# ═══════════════════════════════════════════════════════════════════════════════

def fill_road_mask(closed, contour, contour_area, h, w):
    """
    Draws the contour filled solid white onto a black canvas.
    Result: white = road surface, black = everything else.
    If no valid contour was found, falls back to a full-image white mask.
    """
    road_mask = np.zeros((h, w), dtype=np.uint8)

    if contour is None:
        # Fallback: treat the entire image as road so F4/F5/F6 can still run
        road_mask[:] = 255
        fallback = True
        label    = f"3d FALLBACK — full image used as road mask"
        print("  [3d] Fallback mask — full image = road")
    else:
        cv2.drawContours(road_mask, [contour], -1, 255, thickness=cv2.FILLED)
        fallback = False
        road_pct = 100.0 * cv2.countNonZero(road_mask) / (h * w)
        label    = f"3d Road mask | area={contour_area}px | {road_pct:.1f}% of image"
        print(f"  [3d] Road mask filled — {contour_area} px ({road_pct:.1f}% of frame)")

    return road_mask, to_display(road_mask, label), fallback


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_roi(enhanced, base_name="image"):
    """
    Run all four ROI-extraction steps on one enhanced grayscale image.
    Saves each step output + a 2x3 comparison grid.
    Returns road_mask for use by F4, F5, F6.

    Parameters
    ----------
    enhanced  : grayscale numpy array (480 x 640, uint8) from F1
    base_name : str — image file stem used for output folder naming

    Returns
    -------
    road_mask : binary numpy array (480 x 640, uint8)
                255 = road surface, 0 = non-road background
    """
    h, w = enhanced.shape
    out_dir = os.path.join(RESULTS_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n── Feature 3: {base_name} ──")

    # ── Panel: enhanced grayscale from F1 ─────────────────────────────────────
    enh_disp = to_display(enhanced, "F1 Enhanced (gray) — input")

    # ── Run steps ─────────────────────────────────────────────────────────────
    edges,     disp_edges             = detect_edges(enhanced)
    closed,    disp_closed            = close_edges(edges)
    contour,   contour_area           = find_largest_contour(closed)
    road_mask, disp_mask, fallback    = fill_road_mask(closed, contour, contour_area, h, w)

    # ── Colour overlay: road area tinted green on the enhanced image ───────────
    overlay = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    green_layer = overlay.copy()
    green_layer[road_mask == 255] = (0, 200, 0)
    blended = cv2.addWeighted(overlay, 0.6, green_layer, 0.4, 0)

    road_px  = cv2.countNonZero(road_mask)
    road_pct = 100.0 * road_px / (h * w)
    fb_note  = " [FALLBACK]" if fallback else ""
    put_label(blended, f"3e Overlay | road={road_pct:.1f}%{fb_note}")

    # ── Draw contour outline in red so it is easy to inspect ──────────────────
    contour_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    if contour is not None:
        cv2.drawContours(contour_vis, [contour], -1, (0, 0, 220), 2)
        put_label(contour_vis, f"3c Contour | area={contour_area}px")
    else:
        put_label(contour_vis, "3c No valid contour — fallback")

    # ── Save individual step images ───────────────────────────────────────────
    cv2.imwrite(os.path.join(out_dir, "step3a_canny.jpg"),    disp_edges)
    cv2.imwrite(os.path.join(out_dir, "step3b_closed.jpg"),   disp_closed)
    cv2.imwrite(os.path.join(out_dir, "step3c_contour.jpg"),  contour_vis)
    cv2.imwrite(os.path.join(out_dir, "step3d_mask.jpg"),     disp_mask)
    cv2.imwrite(os.path.join(out_dir, "step3e_overlay.jpg"),  blended)

    # ── Build 2x3 comparison grid ─────────────────────────────────────────────
    row1 = np.hstack([enh_disp,    disp_edges,  disp_closed])
    row2 = np.hstack([contour_vis, disp_mask,   blended])
    grid = np.vstack([row1, row2])
    scale = min(1.0, 1280 / grid.shape[1])
    grid  = cv2.resize(grid, (0, 0), fx=scale, fy=scale)

    grid_path = os.path.join(out_dir, "FEATURE3_grid.jpg")
    cv2.imwrite(grid_path, grid)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"  Road area       : {road_px} px ({road_pct:.1f}% of frame)")
    print(f"  Fallback used   : {fallback}")
    print(f"  Saved to        : {out_dir}/")

    # ── Show grid window ──────────────────────────────────────────────────────
    cv2.imshow(f"Feature 3 — {base_name}", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return road_mask   # pass this into F4, F5, F6


# ═══════════════════════════════════════════════════════════════════════════════
# Inline Feature 1 — used only when running F3 standalone
# ═══════════════════════════════════════════════════════════════════════════════

def _load_and_enhance(image_path):
    """Minimal F1 replication so F3 can run standalone."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return None
    img   = cv2.resize(img, (640, 480))
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh   = clahe.apply(gray)
    table = np.array([(i / 255.0) ** (1 / 1.3) * 255 for i in range(256)], dtype=np.uint8)
    enh   = cv2.LUT(enh, table)
    enh   = cv2.medianBlur(enh, 5)
    return enh


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
        print("  Single image : python feature3_roi_extraction.py path/to/img.jpg")
        print("  Batch        : put images in data/raw/ subfolders")
        sys.exit(1)

    print(f"[INFO] Found {len(image_paths)} image(s) — processing all...\n")
    for img_path in image_paths:
        enh = _load_and_enhance(img_path)
        if enh is None:
            continue
        name = os.path.splitext(os.path.basename(img_path))[0]
        extract_roi(enh, base_name=name)

    print("\n[DONE] Feature 3 complete.")