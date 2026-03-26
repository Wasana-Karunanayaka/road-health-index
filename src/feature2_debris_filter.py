"""
Feature 2 — Colour-Based Debris Filter
Objective 5: Identify mud, vegetation, and debris on road surfaces.

Detects brownish mud and green vegetation using HSV colour thresholding.
Outputs a binary debris mask and a debris coverage ratio (0.0 – 1.0).
The ratio feeds into the RHI formula in Feature 12 with 15% weight.

Steps:
  2a. BGR → HSV conversion       — always applied
  2b. Mud mask (HSV threshold)   — brownish / orange-brown pixel range
  2c. Vegetation mask             — green pixel range
  2d. Combine masks               — bitwise OR of both masks
  2e. Compute debris_ratio        — white pixels / total pixels

Requires Feature 1 outputs:
  image    — original BGR image  (needed for HSV colour work)
  enhanced — grayscale processed image (passed through; used by F3+)

Run:  python feature2_debris_filter.py path/to/image.jpg
      python feature2_debris_filter.py              <- batch over data/raw/
Output: data/results/feature2/
"""

import cv2
import numpy as np
import os
import sys
import glob

# ── Folders ────────────────────────────────────────────────────────────────────
RAW_DIR     = "data/raw"
RESULTS_DIR = "data/results/feature2"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── HSV colour ranges ─────────────────────────────────────────────────────────
# OpenCV HSV:  H = 0–179,  S = 0–255,  V = 0–255
MUD_LOWER   = np.array([10,  40,  40])   # brownish mud lower bound
MUD_UPPER   = np.array([30, 255, 200])   # brownish mud upper bound
GREEN_LOWER = np.array([35,  40,  40])   # vegetation green lower bound
GREEN_UPPER = np.array([85, 255, 255])   # vegetation green upper bound

# ── Label drawing (identical style to Feature 1) ──────────────────────────────
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


def mask_to_display(mask, label):
    """Convert binary mask to BGR and add a label — for saving / showing."""
    d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    put_label(d, label)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2a — BGR to HSV
# ═══════════════════════════════════════════════════════════════════════════════

def convert_to_hsv(image):
    """
    Convert BGR to HSV colour space.
    HSV separates colour (Hue) from brightness (Value), so the same mud
    in shadow and in sunlight shares the same Hue range — stable filtering.
    """
    hsv     = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)   # re-render for display
    put_label(hsv_vis, "2a HSV conversion")
    print("  [2a] BGR → HSV done")
    return hsv, hsv_vis


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2b — Mud mask
# ═══════════════════════════════════════════════════════════════════════════════

def create_mud_mask(hsv):
    """
    Threshold for brownish / orange-brown mud tones (H: 10–30).
    Pixels in the mud HSV range → white (255) in mud_mask.
    """
    mud_mask = cv2.inRange(hsv, MUD_LOWER, MUD_UPPER)
    mud_px   = cv2.countNonZero(mud_mask)
    mud_pct  = 100.0 * mud_px / mud_mask.size
    label    = f"2b Mud mask | {mud_pct:.1f}% | H:{MUD_LOWER[0]}-{MUD_UPPER[0]}"
    print(f"  [2b] Mud mask — {mud_px} px ({mud_pct:.1f}%)")
    return mud_mask, mask_to_display(mud_mask, label), mud_pct


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2c — Vegetation mask
# ═══════════════════════════════════════════════════════════════════════════════

def create_vegetation_mask(hsv):
    """
    Threshold for green vegetation tones (H: 35–85).
    Catches grass, leaves, and uprooted plant material on the road surface.
    """
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    green_px   = cv2.countNonZero(green_mask)
    green_pct  = 100.0 * green_px / green_mask.size
    label      = f"2c Vegetation mask | {green_pct:.1f}% | H:{GREEN_LOWER[0]}-{GREEN_UPPER[0]}"
    print(f"  [2c] Vegetation mask — {green_px} px ({green_pct:.1f}%)")
    return green_mask, mask_to_display(green_mask, label), green_pct


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2d — Combine masks
# ═══════════════════════════════════════════════════════════════════════════════

def combine_masks(mud_mask, green_mask):
    """
    Merge mud and vegetation masks with bitwise OR.
    Any pixel flagged by either mask becomes 255 in debris_mask.
    """
    debris_mask = cv2.bitwise_or(mud_mask, green_mask)
    debris_px   = cv2.countNonZero(debris_mask)
    overlap_px  = cv2.countNonZero(cv2.bitwise_and(mud_mask, green_mask))
    debris_pct  = 100.0 * debris_px / debris_mask.size
    label       = f"2d Combined | debris={debris_pct:.1f}% | overlap={overlap_px}px"
    print(f"  [2d] Masks combined — {debris_px} debris px, {overlap_px} overlap px")
    return debris_mask, mask_to_display(debris_mask, label)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2e — Debris coverage ratio
# ═══════════════════════════════════════════════════════════════════════════════

def compute_debris_ratio(debris_mask):
    """
    debris_ratio = debris pixels / total pixels.
    Range: 0.0 (no debris) → 1.0 (fully covered).
    Feeds into the RHI formula (Feature 12) at 15% weight.
    """
    debris_ratio = cv2.countNonZero(debris_mask) / debris_mask.size
    print(f"  [2e] debris_ratio = {debris_ratio:.4f}  ({100*debris_ratio:.1f}% of image)")
    return debris_ratio


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def filter_debris(image, enhanced, base_name="image"):
    """
    Run all debris-filter steps on one image.
    Saves each step output + a 2x3 comparison grid.
    Returns debris_mask and debris_ratio for use by the pipeline.

    Parameters
    ----------
    image     : BGR numpy array (H x W x 3) — original colour image from F1
    enhanced  : grayscale numpy array (H x W) — processed image from F1
    base_name : str — image file stem used for output folder naming

    Returns
    -------
    debris_mask  : binary image (H x W) uint8 — white=debris, black=clear
    debris_ratio : float 0.0–1.0 — fraction of image covered by debris
    """
    out_dir = os.path.join(RESULTS_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n── Feature 2: {base_name} ──")

    # ── Panel: original colour ────────────────────────────────────────────────
    orig_disp = image.copy()
    put_label(orig_disp, "Original (colour)")

    # ── Panel: enhanced grayscale from F1 ─────────────────────────────────────
    enh_disp = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    put_label(enh_disp, "F1 Enhanced (gray) — input")

    # ── Run steps ─────────────────────────────────────────────────────────────
    hsv,        disp_hsv                = convert_to_hsv(image)
    mud_mask,   disp_mud,   mud_pct     = create_mud_mask(hsv)
    green_mask, disp_green, green_pct   = create_vegetation_mask(hsv)
    debris_mask, disp_combined          = combine_masks(mud_mask, green_mask)
    debris_ratio                        = compute_debris_ratio(debris_mask)

    # ── Coloured overlay: debris pixels in red on the original image ──────────
    overlay = image.copy()
    overlay[debris_mask == 255] = (0, 0, 220)           # red over debris areas
    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    debris_pct = 100.0 * debris_ratio
    put_label(blended, f"2e Overlay | debris={debris_pct:.1f}% | ratio={debris_ratio:.4f}")
    put_label(blended, f"mud={mud_pct:.1f}%  veg={green_pct:.1f}%", pos=(10, 50))

    # ── Save individual step images ───────────────────────────────────────────
    cv2.imwrite(os.path.join(out_dir, "step2a_hsv.jpg"),        disp_hsv)
    cv2.imwrite(os.path.join(out_dir, "step2b_mud.jpg"),        disp_mud)
    cv2.imwrite(os.path.join(out_dir, "step2c_vegetation.jpg"), disp_green)
    cv2.imwrite(os.path.join(out_dir, "step2d_combined.jpg"),   disp_combined)
    cv2.imwrite(os.path.join(out_dir, "step2e_overlay.jpg"),    blended)

    # ── Build 2x3 comparison grid ─────────────────────────────────────────────
    row1 = np.hstack([orig_disp,  enh_disp,   disp_hsv])
    row2 = np.hstack([disp_mud,   disp_green, blended])
    grid = np.vstack([row1, row2])
    scale = min(1.0, 1280 / grid.shape[1])
    grid  = cv2.resize(grid, (0, 0), fx=scale, fy=scale)

    grid_path = os.path.join(out_dir, "FEATURE2_grid.jpg")
    cv2.imwrite(grid_path, grid)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"  Mud coverage    : {mud_pct:.1f}%")
    print(f"  Vegetation      : {green_pct:.1f}%")
    print(f"  Total debris    : {debris_pct:.1f}%  (ratio = {debris_ratio:.4f})")
    print(f"  Saved to        : {out_dir}/")

    # ── Show grid window ──────────────────────────────────────────────────────
    cv2.imshow(f"Feature 2 — {base_name}", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return debris_mask, debris_ratio


# ═══════════════════════════════════════════════════════════════════════════════
# Inline Feature 1 — used only when running F2 standalone
# ═══════════════════════════════════════════════════════════════════════════════

def _load_and_enhance(image_path):
    """Minimal F1 replication so F2 can run standalone without importing F1."""
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
    return img, enh


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
        print("  Single image : python feature2_debris_filter.py path/to/img.jpg")
        print("  Batch        : put images in data/raw/ subfolders")
        sys.exit(1)

    print(f"[INFO] Found {len(image_paths)} image(s) — processing all...\n")
    for img_path in image_paths:
        bgr, enh = _load_and_enhance(img_path)
        if bgr is None:
            continue
        name = os.path.splitext(os.path.basename(img_path))[0]
        filter_debris(bgr, enh, base_name=name)

    print("\n[DONE] Feature 2 complete.")