"""
Feature 1 — Adaptive Disaster-Aware Image Enhancement
Objective 1: Enhance images affected by noise, poor illumination, and glare.

Each enhancement step checks whether it is actually needed before applying.
A well-lit, low-noise image passes through mostly unchanged.
A dark, noisy image gets the corrections it needs.

Steps:
  1a. Grayscale conversion        — always applied
  1b. CLAHE                       — only if contrast (std) is below threshold
  1c. Gamma correction            — only if image is too dark OR too bright
  1d. Median filter               — only if salt-and-pepper noise is detected

Run:  python feature1_enhancement.py path/to/image.jpg
      python feature1_enhancement.py              <- runs on all images in data/raw/
Output: data/results/feature1/
"""

import cv2
import numpy as np
import os
import sys
import glob

# ── Folders ────────────────────────────────────────────────────────────────────
RAW_DIR     = "data/raw"
RESULTS_DIR = "data/results/feature1"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Thresholds that decide whether each step is needed ────────────────────────
CLAHE_STD_THRESHOLD    = 45    # apply CLAHE if pixel std deviation is below this
GAMMA_DARK_THRESHOLD   = 85    # apply brightening if mean brightness is below this
GAMMA_BRIGHT_THRESHOLD = 190   # apply darkening  if mean brightness is above this
GAMMA_DARK_VALUE       = 1.4   # gamma when image is too dark  (> 1 = brighter)
GAMMA_BRIGHT_VALUE     = 0.7   # gamma when image is too bright(< 1 = darker)
NOISE_THRESHOLD        = 0.015 # apply median filter if noise pixel fraction exceeds this

# ── Label drawing ──────────────────────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.52
FONT_COLOR = (255, 255, 0)   # yellow
FONT_THICK = 2


def put_label(img, text, pos=(10, 26)):
    """Draw yellow text with a black background so it is readable on any image."""
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    x, y = pos
    cv2.rectangle(img, (x - 3, y - th - 5), (x + tw + 3, y + 3), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, FONT_COLOR, FONT_THICK)
    return img


def to_display(gray, label):
    """Convert grayscale to BGR and add a label — used for saving/showing."""
    d = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    put_label(d, label)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Load and resize
# ═══════════════════════════════════════════════════════════════════════════════

def load_image(image_path):
    """Load image from disk and resize to 640x480 so all kernel sizes stay valid."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return None
    img = cv2.resize(img, (640, 480))
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1a — Grayscale (always applied)
# ═══════════════════════════════════════════════════════════════════════════════

def convert_to_grayscale(image):
    """
    Convert BGR to grayscale.
    Always applied: crack and damage detection uses intensity contrast, not colour.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = int(np.mean(gray))
    std  = int(np.std(gray))
    label = f"1a Grayscale | mean={mean}  std={std}"
    print(f"  [1a] Grayscale done — mean brightness={mean}, contrast std={std}")
    return gray, to_display(gray, label)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1b — CLAHE (only if contrast is too low)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_clahe(gray):
    """
    CLAHE: enhances local contrast by equalising small tiles (8x8) separately.
    Skipped if the image already has good contrast (std >= CLAHE_STD_THRESHOLD).
    Prevents over-brightening images that are already well-lit.
    """
    std = int(np.std(gray))

    if std >= CLAHE_STD_THRESHOLD:
        label = f"1b CLAHE SKIPPED | std={std} >= {CLAHE_STD_THRESHOLD} (already ok)"
        print(f"  [1b] CLAHE skipped — contrast std={std} is good enough")
        return gray, to_display(gray, label), False

    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(gray)
    std_after = int(np.std(enhanced))
    label = f"1b CLAHE APPLIED | std: {std} -> {std_after}"
    print(f"  [1b] CLAHE applied — contrast std: {std} -> {std_after}")
    return enhanced, to_display(enhanced, label), True


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1c — Gamma correction (only if image is too dark or too bright)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_gamma_correction(gray):
    """
    Gamma < 1 darkens, gamma > 1 brightens.
    Skipped if mean brightness is already in a normal range.
    Prevents washing out images that are already correctly exposed.
    """
    mean = int(np.mean(gray))

    if GAMMA_DARK_THRESHOLD <= mean <= GAMMA_BRIGHT_THRESHOLD:
        label = f"1c Gamma SKIPPED | mean={mean} in normal range [{GAMMA_DARK_THRESHOLD}-{GAMMA_BRIGHT_THRESHOLD}]"
        print(f"  [1c] Gamma skipped — mean={mean} is in the normal range")
        return gray, to_display(gray, label), 1.0

    if mean < GAMMA_DARK_THRESHOLD:
        gamma  = GAMMA_DARK_VALUE
        reason = f"too dark (mean={mean})"
    else:
        gamma  = GAMMA_BRIGHT_VALUE
        reason = f"too bright (mean={mean})"

    inv_gamma = 1.0 / gamma
    lut = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8
    )
    corrected  = cv2.LUT(gray, lut)
    mean_after = int(np.mean(corrected))
    label = f"1c Gamma={gamma} ({reason}) | mean: {mean} -> {mean_after}"
    print(f"  [1c] Gamma={gamma} applied ({reason}) — mean: {mean} -> {mean_after}")
    return corrected, to_display(corrected, label), gamma


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1d — Median filter (only if noise is detected)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_median_filter(gray):
    """
    Removes salt-and-pepper noise (random bright/dark specks).
    Skipped if the fraction of outlier pixels is below NOISE_THRESHOLD.
    Skipping preserves fine crack detail on already-clean images.
    """
    total_px    = gray.size
    outlier_px  = int(np.sum((gray < 15) | (gray > 240)))
    noise_ratio = outlier_px / total_px

    if noise_ratio < NOISE_THRESHOLD:
        label = f"1d Median SKIPPED | noise={noise_ratio:.4f} < {NOISE_THRESHOLD}"
        print(f"  [1d] Median filter skipped — noise ratio={noise_ratio:.4f} is low")
        return gray, to_display(gray, label), False

    denoised      = cv2.medianBlur(gray, 5)
    outlier_after = int(np.sum((denoised < 15) | (denoised > 240)))
    noise_after   = outlier_after / total_px
    label = f"1d Median APPLIED | noise: {noise_ratio:.4f} -> {noise_after:.4f}"
    print(f"  [1d] Median applied — noise ratio: {noise_ratio:.4f} -> {noise_after:.4f}")
    return denoised, to_display(denoised, label), True


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def enhance_image(image_path):
    """
    Run all four adaptive enhancement steps on one image.
    Saves each step output + a 2x3 comparison grid.
    Returns the final enhanced grayscale image for use by Feature 2+.
    """
    original = load_image(image_path)
    if original is None:
        return None

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir   = os.path.join(RESULTS_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n── Feature 1: {base_name} ──")

    # ── Original display panel ────────────────────────────────────────────────
    orig_disp = original.copy()
    put_label(orig_disp, "Original (colour)")

    # ── Run steps ─────────────────────────────────────────────────────────────
    gray,   disp_gray                     = convert_to_grayscale(original)
    clahe_, disp_clahe, clahe_applied     = apply_clahe(gray)
    gamma_, disp_gamma, gamma_val         = apply_gamma_correction(clahe_)
    final,  disp_final, median_applied    = apply_median_filter(gamma_)

    # ── Save each step ────────────────────────────────────────────────────────
    cv2.imwrite(os.path.join(out_dir, "step1a_grayscale.jpg"), disp_gray)
    cv2.imwrite(os.path.join(out_dir, "step1b_clahe.jpg"),     disp_clahe)
    cv2.imwrite(os.path.join(out_dir, "step1c_gamma.jpg"),     disp_gamma)
    cv2.imwrite(os.path.join(out_dir, "step1d_median.jpg"),    disp_final)

    # ── Difference image (shows what changed overall) ─────────────────────────
    diff         = cv2.absdiff(gray, final)
    diff_bright  = cv2.convertScaleAbs(diff, alpha=6)  # amplify so small changes are visible
    diff_display = cv2.cvtColor(diff_bright, cv2.COLOR_GRAY2BGR)
    changed_px   = int(np.sum(diff > 5))
    put_label(diff_display, f"Diff (x6) | changed px={changed_px}")

    # ── Annotate final panel with what was applied ────────────────────────────
    steps_applied = []
    if clahe_applied:    steps_applied.append("CLAHE")
    if gamma_val != 1.0: steps_applied.append(f"Gamma={gamma_val}")
    if median_applied:   steps_applied.append("Median")
    if not steps_applied:
        steps_applied = ["None (image was fine)"]

    final_disp = disp_final.copy()
    put_label(final_disp, f"Final | Applied: {', '.join(steps_applied)}", pos=(10, 26))
    put_label(final_disp, f"mean={int(np.mean(final))}  std={int(np.std(final))}", pos=(10, 50))

    # ── Build 2x3 comparison grid ─────────────────────────────────────────────
    row1 = np.hstack([orig_disp, disp_gray, disp_clahe])
    row2 = np.hstack([disp_gamma, final_disp, diff_display])
    grid = np.vstack([row1, row2])
    scale = min(1.0, 1280 / grid.shape[1])  # shrink to fit screen
    grid  = cv2.resize(grid, (0, 0), fx=scale, fy=scale)

    grid_path = os.path.join(out_dir, "FEATURE1_grid.jpg")
    cv2.imwrite(grid_path, grid)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"  Steps applied  : {', '.join(steps_applied)}")
    print(f"  Final mean     : {int(np.mean(final))}   std: {int(np.std(final))}")
    print(f"  Saved to       : {out_dir}/")

    # ── Show grid window ──────────────────────────────────────────────────────
    cv2.imshow(f"Feature 1 — {base_name}", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return final   # pass this grayscale image into Feature 2


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point — single image or batch over data/raw/
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        enhance_image(sys.argv[1])
    else:
        # Find all images under data/raw/ (rdd2022, crackforest, rescuenet subfolders)
        patterns = [
            os.path.join(RAW_DIR, "**", "*.jpg"),
            os.path.join(RAW_DIR, "**", "*.png"),
            os.path.join(RAW_DIR, "**", "*.jpeg"),
        ]
        images = []
        for p in patterns:
            images.extend(glob.glob(p, recursive=True))

        if not images:
            print(f"[ERROR] No images found under '{RAW_DIR}/'")
            print("  Pass a single image:  python feature1_enhancement.py path/to/img.jpg")
            print("  Or put images in  :   data/raw/rdd2022/   data/raw/crackforest/   etc.")
            sys.exit(1)

        print(f"[INFO] Found {len(images)} image(s) — processing all...\n")
        for img_path in images:
            enhance_image(img_path)

    print("\n[DONE] Feature 1 complete.")