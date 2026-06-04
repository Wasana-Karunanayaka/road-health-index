"""
feature3_roi_extraction.py
--------------------------
PURPOSE : Extract road surface Region of Interest (ROI) as a binary mask.
INPUTS  : enhanced (gray ndarray 480x640), profile (dict from F1)
OUTPUTS : road_mask (binary ndarray 480x640 — 255=road / 0=background)
FEEDS   : F4 (glare suppression), F5 (crack detection), F6 (pothole detection),
          F8 (crack metrics), F9 (texture), F10 (edge fragmentation)
"""

import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import save_result, make_grid

# ── Tunable Parameters ─────────────────────────────────────────────────────────
CANNY_LOW       = 50    # weak edge threshold — kept only if connected to a strong edge
CANNY_HIGH      = 150   # strong edge threshold — always kept regardless of neighbours
CLOSE_KERNEL_SZ = 15    # closing kernel size — bridges gaps up to ~7 px in road boundary
MIN_ROAD_AREA   = 5000  # minimum contour area (px²) to be treated as a valid road region


def extract_roi(enhanced, profile, base_name='image'):
    """
    Extract road surface as a binary mask: Canny edges → closing → largest contour fill.
    Falls back to full-image mask for aerial images where contour logic breaks down.

    Args:
        enhanced  : ndarray (480×640, uint8) — grayscale enhanced image from F1
        profile   : dict  — image condition flags from F1 (is_aerial, is_dark, ...)
        base_name : str   — image stem for save_result file naming

    Returns:
        road_mask : ndarray (480×640, uint8) — binary, 255=road / 0=background
        (Step 3b still returns closed edges as placeholder until 3d fills the mask)
    """

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 3a — Canny Edge Detection
    # ══════════════════════════════════════════════════════════════════════════════


    # WHY CANNY NOT SOBEL/SCHARR: Canny adds two stages Sobel/Scharr lack.
    # Non-maximum suppression thins every edge ridge to exactly 1 pixel wide —
    # without this, thick gradient bands produce multiple overlapping contours
    # and cv2.findContours in step 3c picks the wrong one.
    # Hysteresis thresholding uses TWO thresholds: pixels above CANNY_HIGH are
    # definite edges; pixels between CANNY_LOW and CANNY_HIGH are kept only if
    # directly connected to a definite-edge pixel. This preserves the full
    # continuous road boundary loop while dropping isolated noise speckles.

    # WHY THESE VALUES: low=50 / high=150 is intentionally wide (3:1 ratio).
    # A narrow ratio (e.g. 100/150) would break the road boundary at low-contrast
    # sections (wet patches, shadow transitions) — step 3b closing can bridge
    # small gaps, but only if the majority of the boundary edge is present first.
    edges = cv2.Canny(enhanced, CANNY_LOW, CANNY_HIGH)
    save_result(edges, 3, base_name, 'step3a_canny')


    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 3b — Morphological Closing (bridge boundary gaps)
    # ══════════════════════════════════════════════════════════════════════════════


    # WHY CLOSING NOT JUST DILATION: Pure dilation would permanently thicken every
    # edge — a 15×15 dilation turns 1-px lines into 15-px bands, making the contour
    # border inaccurate and causing cv2.fillPoly to over-fill into non-road regions.
    # Closing (dilate → erode) bridges the gap but then pulls the edges back to
    # approximately their original 1-px width, preserving contour shape accuracy.

    # WHY 15×15: The kernel radius (~7 px) must exceed the widest expected boundary
    # gap. Common gap sources on road photos:
    #   - Low-contrast wet patches where road surface matches verge brightness (~5 px)
    #   - Shadow edges crossing the road boundary and interrupting it (~4 px)
    #   - JPEG blocking artefacts at pavement-to-grass transitions (~3 px)
    # 15×15 covers all of these. Larger (e.g. 25×25) risks merging separate contours;
    # smaller (e.g. 9×9) leaves gaps that break the closed loop for findContours.

    # WHY RECT NOT ELLIPSE: Road boundaries are mostly horizontal/vertical straight
    # runs (top edge, side kerbs). A rectangular kernel closes these axis-aligned
    # gaps more reliably than an ellipse, which under-closes at corner pixels.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (CLOSE_KERNEL_SZ, CLOSE_KERNEL_SZ)
    )
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)

    # Compare with step3a_canny: gaps in the road boundary line should now be
    # filled. Lane marking gaps, kerb interruptions, and shadow breaks all close.
    save_result(closed, 3, base_name, 'step3b_closed')


    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 3c — Find Contours → Select Largest (bottom-half bias)
    # ══════════════════════════════════════════════════════════════════════════════

    # WHY LARGEST CONTOUR: Road boundary = dominant closed loop after closing.
    # Any other contours (lane markings, objects, shadow patches) are smaller.

    # WHY RETR_EXTERNAL / CHAIN_APPROX_SIMPLE: outer boundary only, compressed
    # straight runs to endpoints — lower memory, faster fill, identical shape.
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter out noise loops smaller than MIN_ROAD_AREA
    valid = [c for c in contours if cv2.contourArea(c) >= MIN_ROAD_AREA]

    # BOTTOM-HALF BIAS: In every ground-level road photo the road surface sits in
    # the lower half of the frame. Vegetation canopy, sky, and vehicles sit in the
    # upper half. Without this bias, a large vegetation or car contour in the upper
    # half can win the max-area selection even though the road is clearly below it.
    # Fix: prefer contours whose centroid y > image_height/2.
    # Centroid y = m01 / m00 from raw image moments (pixel-count weighted average).
    # Fallback: if no valid contour has centroid in the bottom half (aerial / unusual
    # framing), use the unrestricted largest — preserving existing correct behaviour.
    h = enhanced.shape[0]   # 480 px
    bottom = [c for c in valid
              if (cv2.moments(c)['m01'] / (cv2.moments(c)['m00'] + 1e-6)) > h / 2]
    candidates = bottom if bottom else valid
    largest    = max(candidates, key=cv2.contourArea) if candidates else None

    # Draw selected contour outline on enhanced image for visual QA
    contour_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    if largest is not None:
        cv2.drawContours(contour_vis, [largest], -1, (0, 255, 0), 2)
    save_result(contour_vis, 3, base_name, 'step3c_contour')


    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 3d — Fill Contour → Solid road_mask + Aerial Fallback + Summary Grid
    # ══════════════════════════════════════════════════════════════════════════════

    # WHY FILLED NOT OUTLINE: Downstream features call
    # cv2.bitwise_and(img, img, mask=road_mask) to restrict analysis to road pixels.
    # This requires every interior pixel to be 255 — an outline-only mask would
    # pass only the boundary ring itself, masking out the actual road surface.

    # WHY ZEROS_LIKE NOT FULL COPY: We start from a blank canvas so only the
    # interior fill region is 255. Any prior edge/noise pixels in `closed` are
    # excluded — the mask is strictly the filled polygon, nothing else.

    road_mask = np.zeros_like(enhanced)

    if largest is not None:
        # Fill interior of largest contour solid white — this IS the road surface
        cv2.drawContours(road_mask, [largest], -1, 255, thickness=cv2.FILLED)
    else:
        # FALLBACK: no valid road contour found — treat entire image as road.
        # Triggered by: aerial narrow-strip roads (RescueNet), very dark images
        # where Canny finds no dominant boundary, or heavily debris-covered roads.
        # Setting is_aerial=True signals F5/F6 to use larger minimum region areas
        # and F3's own contour failure is documented in the flag for the report.
        road_mask[:] = 255
        profile['is_aerial'] = True   # downstream features adapt their parameters

    # Road coverage: what fraction of the image is inside the road mask.
    # Healthy ground-level photo: 40–80%. Very low (<20%) = likely fallback needed.
    # Aerial image (full-frame fallback): reports 100%.
    road_coverage = cv2.countNonZero(road_mask) / road_mask.size

    # Overlay mask as green tint on original for visual QA
    overlay      = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    green_layer  = np.zeros_like(overlay)
    green_layer[road_mask == 255] = (0, 180, 0)
    overlay      = cv2.addWeighted(overlay, 0.7, green_layer, 0.3, 0)
    save_result(overlay,   3, base_name, 'step3d_road_mask_overlay')
    save_result(road_mask, 3, base_name, 'step3d_road_mask')

    # F3 summary grid — all 4 stages side by side for the report
    grid = make_grid(
        [cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
         cv2.cvtColor(edges,    cv2.COLOR_GRAY2BGR),
         cv2.cvtColor(closed,   cv2.COLOR_GRAY2BGR),
         contour_vis,
         cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR),
         overlay],
        labels=['original', 'canny', 'closed', 'contour', 'mask', 'overlay'],
        cols=3
    )
    save_result(grid, 3, base_name, 'F3_grid')

    return road_mask   




# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST  
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from feature1_enhancement import enhance_image
    from utils import get_all_test_images

    paths = [(sys.argv[1], 'manual')] if len(sys.argv) > 1 \
            else get_all_test_images('data/raw')

    passed = fallback = check = 0
    for img_path, dataset in paths:
        base = os.path.splitext(os.path.basename(img_path))[0]

        _, enhanced, profile = enhance_image(img_path)
        road_mask = extract_roi(enhanced, profile, base_name=base)

        road_px     = cv2.countNonZero(road_mask)
        coverage    = road_px / road_mask.size
        is_fallback = profile.get('is_aerial', False)

        # Re-derive centroid of selected contour to confirm bottom-half bias worked
        edges  = cv2.Canny(enhanced, CANNY_LOW, CANNY_HIGH)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (CLOSE_KERNEL_SZ, CLOSE_KERNEL_SZ))
        closed    = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid  = [c for c in contours if cv2.contourArea(c) >= MIN_ROAD_AREA]
        h      = enhanced.shape[0]
        bottom = [c for c in valid
                  if (cv2.moments(c)['m01'] /
                      (cv2.moments(c)['m00'] + 1e-6)) > h / 2]
        used_bottom = len(bottom) > 0 and not is_fallback

        if is_fallback:
            flag = 'FALLBACK'; fallback += 1
        elif 0.20 <= coverage <= 0.85:
            flag = 'PASS';     passed  += 1
        else:
            flag = 'CHECK';    check   += 1

        bias_note = 'bottom-bias' if used_bottom else ('fallback' if is_fallback
                                                        else 'no-bottom-contour')
        print(f'[{dataset}] {base:<30} | '
              f'coverage={coverage:.1%}  '
              f'bias={bias_note}  [{flag}]')

    print(f'\nPASS={passed}  CHECK={check}  FALLBACK={fallback}  '
          f'— outputs in data/results/feature3/')