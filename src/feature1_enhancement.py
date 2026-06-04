"""
feature1_enhancement.py
-----------------------
PURPOSE : Standardize every input image before any damage analysis begins.
          Corrects resolution, brightness, contrast, and noise so that all
          downstream features operate on a clean, consistent signal.
INPUTS  : image_path (str) — path to any raw road photograph
OUTPUTS : image    (BGR  ndarray 480×640×3) — colour original, kept for F2/F13
          enhanced (gray ndarray 480×640)   — processed grayscale for F3–F11
          profile  (dict)                   — image-condition flags for F2–F11
FEEDS   : ALL downstream features (F2 – F13)
"""

import cv2
import numpy as np
import os
import sys

# Add src/ to path so utils.py is importable when running this file directly
sys.path.insert(0, os.path.dirname(__file__))
from utils import save_result, compute_stats, profile_image, make_grid

# ── Tunable Parameters ──────────────────────────────────────────────────────────
RESIZE_WH     = (640, 480)    # (width, height) — standard for all pipeline images
                               # 640×480 chosen: fast to process, kernels stay meaningful
CLAHE_TILE    = (8, 8)        # CLAHE tile grid — 8×8 divides 480/640 evenly into 60/80 px tiles
NOISE_THRESH  = 0.015         # >1.5 % of pixels near black/white = salt-and-pepper noise present



def enhance_image(image_path):
    """
    Load a raw road image, standardize its resolution, convert to grayscale,
    and apply adaptive CLAHE to boost local contrast where needed.

    Steps implemented: 1a (load/resize/gray), 1b (adaptive CLAHE).
    Steps 1c–1f (gamma, median, profile finalisation) added in later sessions.

    Parameters
    ----------
    image_path : str   path to input image (JPG, PNG, etc.)

    Returns
    -------
    image    : np.ndarray  BGR  480×640×3  — colour original (kept for F2/F13)
    enhanced : np.ndarray  gray 480×640    — grayscale (used by F3–F11)
    profile  : dict        image-condition flags (populated fully in Step 1f)
    """

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 1a — Load, resize, convert to grayscale
    # ══════════════════════════════════════════════════════════════════════════════


    # ── Step 1a-i: Load image from disk ───────────────────────────────────────
    # cv2.imread returns BGR (not RGB) — all cv2 functions expect BGR ordering.
    # Returns None if the path is wrong; catch this early so the error is clear.
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            f"enhance_image: could not read image at '{image_path}'. "
            "Check the path and file format."
        )

    # ── Step 1a-ii: Resize to standard resolution ─────────────────────────────
    # INTER_AREA is the correct interpolation for downscaling:
    #   - averages pixels in the source region → no aliasing artefacts
    #   - INTER_LINEAR / CUBIC would introduce ringing at sharp crack edges
    # INTER_LINEAR used for upscaling (rarely needed for road datasets).
    h_orig, w_orig = image.shape[:2]
    target_w, target_h = RESIZE_WH   # (640, 480)

    if w_orig > target_w or h_orig > target_h:
        interp = cv2.INTER_AREA       # shrinking → area average
    else:
        interp = cv2.INTER_LINEAR     # growing  → bilinear

    image = cv2.resize(image, (target_w, target_h), interpolation=interp)
    # image.shape is now (480, 640, 3) — confirmed for every downstream kernel

    # ── Step 1a-iii: Convert to grayscale ─────────────────────────────────────
    # Road damage is a light/dark contrast signal, not a colour signal.
    # Carrying BGR through the pipeline triples memory and adds no detection value.
    # cv2.COLOR_BGR2GRAY applies the ITU-R BT.601 luminance formula:
    #   Y = 0.114·B + 0.587·G + 0.299·R
    # This weighting matches human brightness perception and preserves crack contrast.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray.shape is now (480, 640) — single channel, uint8, 0–255

    # ── Step 1a-iv: Profile the raw grayscale image ───────────────────────────
    # profile_image() examines brightness, contrast, wetness, and scene type.
    # Running it on the raw (pre-enhancement) gray gives an honest baseline —
    # enhancement hasn't changed the statistics yet.
    # Downstream features (F3–F6) read these flags to adapt their parameters.
    profile = profile_image(image, gray)

    # enhanced starts as raw gray; Steps 1b–1e modify it in place progressively.
    enhanced = gray.copy()


    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 1b — Adaptive CLAHE — local contrast enhancement
    # ══════════════════════════════════════════════════════════════════════════════


    # CLAHE divides the image into CLAHE_TILE (8×8) tiles and equalizes each
    # independently, so dark corners are boosted without over-brightening bright
    # areas — global histogram equalization would do both at once and destroy the
    # contrast ratio between crack (dark) and road surface (lighter).
    #
    # Applied only when std < 45: a high-std image already has strong local
    # contrast between damage and surface; forcing CLAHE introduces tile-boundary
    # artefacts that look like cracks to F5.
    #
    # clipLimit scales with how poor the contrast is:
    #   std < 30  → very flat image → clip = 3.0  (strongest boost allowed)
    #   std < 45  → low contrast   → clip = 2.0  (moderate boost)
    #   std ≥ 45  → skip entirely  (no CLAHE applied)
    std_val = float(np.std(enhanced))

    if std_val < 45:                           # contrast is poor enough to warrant CLAHE
        clip_lim = 3.0 if std_val < 30 else 2.0
        clahe    = cv2.createCLAHE(
            clipLimit=clip_lim,
            tileGridSize=CLAHE_TILE            # (8,8) tiles over 480×640 → 60×80 px each
        )
        enhanced = clahe.apply(enhanced)
        # enhanced std will have increased; downstream steps see a higher-contrast image


    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 1c — Adaptive Gamma Correction — fix overall brightness
    # ══════════════════════════════════════════════════════════════════════════════


    # Gamma remaps brightness via a power curve: out = (in/255)^(1/γ) × 255
    # γ > 1 brightens (exponent < 1 pulls dark values up toward mid-grey)
    # γ < 1 darkens  (exponent > 1 pushes bright values down)
    # Applied AFTER CLAHE so we correct overall level, not local contrast.
    # Skip if mean is already in the healthy 85–175 range (γ = 1.0 = identity).
    mean_val = float(np.mean(enhanced))

    if   mean_val < 60:   gamma = 1.60   # very dark (night/flood shadow) → strong brighten
    elif mean_val < 85:   gamma = 1.35   # dark → gentle brighten
    elif mean_val > 185:  gamma = 0.70   # washed-out/overexposed → strong darken
    elif mean_val > 155:  gamma = 0.85   # slightly bright → gentle darken
    else:                 gamma = 1.00   # 85–155 range → no change needed

    if gamma != 1.0:
        # Build a 256-entry LUT: index = input value, value = corrected output
        # LUT avoids computing pow() for every pixel — 256 lookups vs 307,200
        lut      = np.array(
            [(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)],
            dtype=np.uint8
        )
        enhanced = cv2.LUT(enhanced, lut)


    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 1d — Median Filter — remove salt-and-pepper noise
    # ══════════════════════════════════════════════════════════════════════════════


    # Salt-and-pepper = isolated pixels at extreme brightness (sensor noise,
    # rain drops, compression artefacts). They produce false crack detections
    # in F5 because they create sharp local contrast exactly like crack edges.
    #
    # Median filter: replaces each pixel with the median of its 5×5 neighbourhood.
    # Unlike Gaussian blur, a single outlier pixel cannot drag the median —
    # it must be the majority value to influence the result.
    # This preserves real crack edges while erasing isolated speckles.
    #
    # Applied only when noise fraction > NOISE_THRESH (1.5 %):
    # unnecessary median filtering marginally blurs thin hairline cracks.
    near_black = int(np.sum(enhanced < 10))
    near_white = int(np.sum(enhanced > 245))
    noise_frac = (near_black + near_white) / float(enhanced.size)

    if noise_frac > NOISE_THRESH:
        enhanced = cv2.medianBlur(enhanced, 5)   # 5×5 kernel — smallest size that
                                                  # clears multi-pixel speckle clusters


    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 1e — Re-profile on the fully enhanced image
    # ══════════════════════════════════════════════════════════════════════════════


    # Steps 1b–1d changed brightness, contrast, and noise — the profile computed
    # in Step 1a on raw gray may now have stale flags (e.g. is_dark was True but
    # gamma correction has brought mean into the normal range).
    # Re-running profile_image ensures F3–F6 receive flags that match the actual
    # enhanced image they will process, not the uncorrected original.
    profile = profile_image(image, enhanced)   # overwrites the Step 1a profile

    return image, enhanced, profile



# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    from utils import get_all_test_images

    # ── Image selection ───────────────────────────────────────────────────────
    # Single image mode:  python feature1_enhancement.py path/to/image.jpg
    # Batch mode:         python feature1_enhancement.py
    if len(sys.argv) > 1:
        test_images = [(sys.argv[1], 'manual')]
    else:
        test_images = get_all_test_images()
        if not test_images:
            print("No images found in data/raw/. Add images and retry.")
            sys.exit(1)

    print(f"\n=== Feature 1 — Steps 1a–1e | {len(test_images)} image(s) ===\n")

    passed = 0
    failed = []

    for img_path, dataset in test_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{dataset}] {base_name} ...", end=' ', flush=True)

        try:
            # ── Run pipeline ──────────────────────────────────────────────────
            raw         = cv2.imread(img_path)
            raw         = cv2.resize(raw, RESIZE_WH, interpolation=cv2.INTER_AREA)
            gray_raw    = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

            # 1b CLAHE intermediate
            std_val = float(np.std(gray_raw))
            if std_val < 45:
                clip_lim    = 3.0 if std_val < 30 else 2.0
                after_clahe = cv2.createCLAHE(clipLimit=clip_lim,
                                               tileGridSize=CLAHE_TILE).apply(gray_raw)
            else:
                after_clahe = gray_raw.copy()

            # 1c gamma intermediate
            mean_val = float(np.mean(after_clahe))
            if   mean_val < 60:   gamma = 1.60
            elif mean_val < 85:   gamma = 1.35
            elif mean_val > 185:  gamma = 0.70
            elif mean_val > 155:  gamma = 0.85
            else:                 gamma = 1.00
            if gamma != 1.0:
                lut         = np.array([(i/255.0)**(1.0/gamma)*255
                                        for i in range(256)], dtype=np.uint8)
                after_gamma = cv2.LUT(after_clahe, lut)
            else:
                after_gamma = after_clahe.copy()

            # 1d median intermediate
            nb = int(np.sum(after_gamma < 10))
            nw = int(np.sum(after_gamma > 245))
            nf = (nb + nw) / float(after_gamma.size)
            after_median = cv2.medianBlur(after_gamma, 5) if nf > NOISE_THRESH else after_gamma.copy()

            # Full pipeline output
            image, enhanced, profile = enhance_image(img_path)

            # ── Save outputs ──────────────────────────────────────────────────
            save_result(image,        1, base_name, 'step1a_original')
            save_result(gray_raw,     1, base_name, 'step1a_gray')
            save_result(after_clahe,  1, base_name, 'step1b_clahe')
            save_result(after_gamma,  1, base_name, 'step1c_gamma')
            save_result(after_median, 1, base_name, 'step1d_median')
            save_result(enhanced,     1, base_name, 'step1e_final')

            grid = make_grid(
                [raw, gray_raw, after_clahe, after_gamma, after_median, enhanced],
                labels=['1a:Original','1a:Gray','1b:CLAHE',
                        '1c:Gamma','1d:Median','1e:Final'],
                cols=3
            )
            save_result(grid, 1, base_name, 'F1_grid')

            # ── Per-image summary line ────────────────────────────────────────
            flags = (f"dark={profile['is_dark']} | wet={profile['is_wet']} | "
                     f"aerial={profile['is_aerial']} | "
                     f"mean={profile['mean']:.0f} std={profile['std']:.0f}")
            print(f"OK  →  {flags}")
            passed += 1

        except Exception as e:
            print(f"FAILED — {e}")
            failed.append((img_path, str(e)))

    # ── Batch summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"Results: {passed}/{len(test_images)} passed")
    if failed:
        print("Failed images:")
        for path, err in failed:
            print(f"  {path} → {err}")
    print(f"Outputs → data/results/feature1/<image_name>/")
    print("=== Feature 1 batch test complete ===\n")