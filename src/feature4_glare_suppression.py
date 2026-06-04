"""
feature4_glare_suppression.py
------------------------------
PURPOSE : Remove false gradient spikes caused by water/flood reflections on road surface.
          Wet-road glare hotspots produce the same sharp brightness transitions as crack
          edges — without suppression, F5 would detect every reflection as a fake crack.
INPUTS  : enhanced  (gray ndarray 480×640)  — from F1
          road_mask (uint8 ndarray 480×640)  — from F3  (255=road, 0=background)
          profile   (dict)                  — from F1  (is_wet, is_very_wet, glare_ratio)
OUTPUTS : suppressed  (uint8 ndarray 480×640) — gradient map with glare regions zeroed
          glare_score (float 0–1)             — fraction of road pixels affected by glare
FEEDS   : F5 crack detection (suppressed gradient used as optional input)
          F12 RHI computation (glare_score at 10% weight)
"""

import cv2
import numpy as np
import os
import sys

# ── Path fix so this file can import utils when run directly ──────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import save_result, compute_stats, make_grid, get_all_test_images

# ── Tunable Parameters ────────────────────────────────────────────────────────
GLARE_DILATE_SZ = 5    # dilation kernel for glare mask — captures bright halo around hotspot
                        # 5×5 extends each flagged pixel ~2 px outward in all directions



def suppress_glare(enhanced, road_mask, profile, base_name):
    """
    Compute Scharr gradient map, suppress glare hotspot regions, return cleaned map.

    Parameters
    ----------
    enhanced  : uint8 ndarray (480, 640)   — grayscale enhanced image from F1
    road_mask : uint8 ndarray (480, 640)   — binary road mask from F3
    profile   : dict                       — image condition flags from F1
    base_name : str                        — image stem used for save_result paths

    Returns
    -------
    suppressed  : uint8 ndarray (480, 640) — gradient magnitude with glare zeroed out
    glare_score : float [0.0, 1.0]         — fraction of road area affected by glare
    """

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 4a — Scharr Gradient
    # ══════════════════════════════════════════════════════════════════════════════


    # Scharr is preferred over Sobel for this task because it uses optimised
    # coefficients ([-3,0,3] / [3,10,3] weighting) that give more accurate gradient
    # magnitude at 45° diagonal orientations.  Road cracks are rarely perfectly
    # horizontal or vertical, so diagonal accuracy directly affects whether a real
    # thin crack edge is captured or missed.
    #
    # Why CV_64F depth:
    #   A gradient can be positive (brightness increasing left→right) or negative
    #   (decreasing).  If we use uint8 output, OpenCV clips all negative values to 0
    #   before we compute magnitude — left-facing and downward-facing edges become
    #   invisible.  CV_64F keeps the full signed range; we square both channels before
    #   taking the root so sign cancels and magnitude is always positive.
    #
    # Why apply road_mask here (not later):
    #   Gradient pixels outside the road boundary carry no useful information for F5
    #   and would dilute the score calculation in Step 4e.  Zeroing them now also
    #   prevents sky/grass gradients from accidentally surviving into the suppressed map.

    sx = cv2.Scharr(enhanced, cv2.CV_64F, 1, 0)   # horizontal gradient  (∂I/∂x)
    sy = cv2.Scharr(enhanced, cv2.CV_64F, 0, 1)   # vertical gradient    (∂I/∂y)

    # Combined gradient magnitude — sqrt(sx² + sy²) gives the strongest directional response
    magnitude_f64 = np.sqrt(sx ** 2 + sy ** 2)

    # Normalise to uint8 [0,255] for consistent downstream thresholding and saving.
    # cv2.normalize with NORM_MINMAX scales the full dynamic range of this image to
    # 0–255 rather than clipping — prevents all gradients being squashed to near-0
    # in low-contrast images where the raw magnitude never reaches large values.
    magnitude_u8 = cv2.normalize(
        magnitude_f64, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Restrict gradient map to road region only — zero all non-road pixels
    magnitude_road = cv2.bitwise_and(magnitude_u8, magnitude_u8, mask=road_mask)

    # Save Step 4a result
    save_result(magnitude_road, 4, base_name, 'step4a_gradient')


    # ══════════════════════════════════════════════════════════════════════
    # STEP 4b — Adaptive Glare Threshold
    # ══════════════════════════════════════════════════════════════════════


    # WHY ADAPTIVE NOT FIXED:
    #   A fixed threshold (e.g. pixel > 230 = glare) breaks on dark images —
    #   no pixel ever reaches 230 so the mask is always empty.  On washed-out
    #   images it over-fires and masks out the entire road surface.
    #   stats['glare_thresh'] = p90 + 0.5*std scales to THIS image's own
    #   brightness distribution, so "glare" always means "unusually bright
    #   relative to this road surface", not an absolute number.
    #
    # WHY profile FLAGS ADJUST IT:
    #   is_very_wet → lower the threshold (many hotspots, need to catch more)
    #   is_wet      → slight reduction (moderate reflections present)
    #   Neither     → use stats value unchanged (dry/normal road)

    stats        = compute_stats(enhanced)
    glare_thresh = stats['glare_thresh']

    if profile.get('is_very_wet'):
        glare_thresh = max(0, glare_thresh - 15)   # aggressive suppression on flooded road
    elif profile.get('is_wet'):
        glare_thresh = max(0, glare_thresh - 8)    # moderate suppression on wet road

    _, glare_mask = cv2.threshold(
        enhanced,
        int(glare_thresh),
        255,
        cv2.THRESH_BINARY        # pixels ABOVE threshold = glare candidate
    )

    # Restrict glare detection to road pixels only — sky/buildings can be bright too
    glare_mask = cv2.bitwise_and(glare_mask, glare_mask, mask=road_mask)

    save_result(glare_mask, 4, base_name, 'step4b_glare_mask')


    # ══════════════════════════════════════════════════════════════════════
    # STEP 4c — Dilate Glare Mask
    # ══════════════════════════════════════════════════════════════════════


    # WHY DILATE THE MASK:
    #   A glare hotspot isn't just the saturated white centre — the bright
    #   halo surrounding it also produces large gradient spikes.  A pixel
    #   just outside the threshold boundary can still be 220–228 brightness,
    #   creating a sharp transition that looks identical to a crack edge.
    #   Dilating by GLARE_DILATE_SZ (5×5, radius ~2 px) extends the masked
    #   zone to cover the full halo, not just the saturated core.
    #
    # WHY 5×5 NOT LARGER:
    #   A 5×5 dilation extends each flagged pixel ~2 px outward.  That is
    #   enough to absorb the halo transition zone.  A larger kernel (e.g.
    #   11×11) would start consuming the edges of legitimate dark cracks
    #   that run close to a reflection — suppressing real damage signal.
    #
    # WHY ELLIPTICAL KERNEL:
    #   Glare hotspots are roughly circular blobs (specular reflections from
    #   a point light source on a flat water surface).  An elliptical kernel
    #   expands them symmetrically in all directions.  A rectangular kernel
    #   would produce square halos with sharp corners — physically wrong and
    #   slightly over-masks diagonal crack edges near hotspot boundaries.

    k_glare    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (GLARE_DILATE_SZ, GLARE_DILATE_SZ))
    glare_mask = cv2.dilate(glare_mask, k_glare, iterations=1)

    save_result(glare_mask, 4, base_name, 'step4c_glare_dilated')


    # ══════════════════════════════════════════════════════════════════════
    # STEP 4d — Zero Glare Regions from Gradient
    # ══════════════════════════════════════════════════════════════════════


    # WHY ZERO NOT BLUR:
    #   We want complete removal of the false gradient signal, not a softer
    #   version of it.  Blurring would still leave a residual spike that
    #   could survive F5's threshold.  Setting to 0 means the glare pixel
    #   contributes nothing to crack detection — it becomes indistinguishable
    #   from a smooth undamaged road patch.
    #
    # WHY USE np.where NOT bitwise_and:
    #   bitwise_and with an inverted mask would work but requires a type
    #   cast round-trip.  np.where is cleaner here: "keep magnitude_road
    #   value where glare_mask is 0, set to 0 where glare_mask is 255".
    #   Single operation, no intermediate array, same result.
    #
    # WHY OPERATE ON magnitude_road NOT enhanced:
    #   We are not modifying the image — we are modifying the GRADIENT MAP.
    #   enhanced stays untouched so F5 can still use it for its adaptive
    #   threshold method (Method A), which needs the original pixel values.
    #   suppressed is a separate artifact — the cleaned gradient only.

    suppressed = np.where(glare_mask == 255, 0, magnitude_road).astype(np.uint8)

    save_result(suppressed, 4, base_name, 'step4d_suppressed')

    
    # ══════════════════════════════════════════════════════════════════════
    # STEP 4e — Glare Score + Summary Grid + Final Return
    # ══════════════════════════════════════════════════════════════════════


    # GLARE SCORE FORMULA:
    #   glare_score = glare pixels inside road / total road pixels
    #   Uses the DILATED mask (step4c) not the raw threshold mask (step4b)
    #   because the dilated zone is what was actually suppressed — that is
    #   the true fraction of road gradient that was unusable due to glare.
    #   Clamped to [0,1] via min() as a safety net against rounding edge cases.
    #
    # WHY ROAD PIXELS AS DENOMINATOR NOT TOTAL IMAGE:
    #   Sky and vegetation can be very bright and would inflate the score if
    #   counted.  road_mask was already applied to glare_mask in Step 4b so
    #   only genuine road glare pixels are in the numerator — denominator
    #   must match to keep the ratio meaningful.
    #
    # SUMMARY GRID — 6 tiles:
    #   original gray | road mask | gradient | glare mask raw |
    #   glare mask dilated | suppressed gradient
    #   Side-by-side comparison makes it immediately obvious whether glare
    #   suppression landed on actual reflections or consumed crack edges.

    road_px     = max(1, int(np.sum(road_mask == 255)))
    glare_px    = int(np.sum(glare_mask == 255))
    glare_score = min(1.0, glare_px / road_px)

    # Reload raw glare mask for grid (step4b output — before dilation)
    raw_glare_path = f'data/results/feature4/{base_name}/step4b_glare_mask.jpg'
    raw_glare      = cv2.imread(raw_glare_path, cv2.IMREAD_GRAYSCALE)
    if raw_glare is None:
        raw_glare = np.zeros_like(enhanced)   # fallback if save failed

    grid = make_grid(
        [enhanced, road_mask, magnitude_road, raw_glare, glare_mask, suppressed],
        labels=['original', 'road_mask', 'gradient',
                'glare_raw', 'glare_dilated', 'suppressed'],
        cols=3
    )
    save_result(grid, 4, base_name, 'F4_grid')

    return suppressed, glare_score




# ══════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    """
    Single image : python feature4_glare_suppression.py path/to/image.jpg
    Batch mode   : python feature4_glare_suppression.py
    """
    from feature1_enhancement import enhance_image
    from feature3_roi_extraction import extract_roi

    if len(sys.argv) > 1:
        test_images = [(sys.argv[1], 'manual')]
    else:
        test_images = get_all_test_images('data/raw')

    pass_count = 0
    fail_count = 0

    for img_path, dataset in test_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f'[{dataset}] {base_name} ... ', end='', flush=True)

        try:
            image, enhanced, profile  = enhance_image(img_path)
            road_mask                 = extract_roi(enhanced, profile, base_name)
            suppressed, glare_score   = suppress_glare(enhanced, road_mask, profile, base_name)
            print(f'OK — glare_score={glare_score:.4f}  '
                  f'wet={profile.get("is_wet")}  '
                  f'very_wet={profile.get("is_very_wet")}')
            pass_count += 1
        except Exception as exc:
            print(f'FAIL — {exc}')
            fail_count += 1

    print(f'\nDone. Passed: {pass_count}  Failed: {fail_count}')