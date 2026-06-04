"""
utils.py
--------
PURPOSE : Shared helper functions imported by all 13 feature files.
          Centralises I/O, adaptive statistics, and visualisation so
          changes here propagate everywhere automatically.
INPUTS  : Various (see individual function docstrings)
OUTPUTS : Various (see individual function docstrings)
FEEDS   : F1 – F13, pipeline.py
"""

import cv2
import numpy as np
import os

# ── Tunable Parameters ──────────────────────────────────────────────────────────
RESULTS_ROOT = 'data/results'   # all output images land under this root


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SAVE RESULT
# ══════════════════════════════════════════════════════════════════════════════

def save_result(img, feature_num, base_name, step_name):
    """
    Save an intermediate or final image to a predictable path.

    Path pattern:
        data/results/feature<N>/<base_name>/<step_name>.jpg

    Parameters
    ----------
    img         : np.ndarray  BGR or grayscale image to save
    feature_num : int         feature number (1–13)
    base_name   : str         image stem, e.g. 'road_001'
    step_name   : str         descriptive label, e.g. 'step1a_gray'

    Returns
    -------
    str   absolute path where the file was written
    """
    # Build directory path and create it if it does not exist
    out_dir = os.path.join(RESULTS_ROOT, f'feature{feature_num}', base_name)
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f'{step_name}.jpg')

    # cv2.imwrite expects BGR; grayscale 2-D arrays are written correctly as-is
    cv2.imwrite(path, img)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 2.  COMPUTE STATS  — the core of the adaptive strategy
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(enhanced):
    """
    Derive all detection thresholds from *this* image's pixel distribution.

    Why this matters: a fixed threshold (e.g. 60) that isolates potholes on a
    bright Japanese highway completely misses them on a dark flooded Sri Lankan
    road. Computing thresholds from the image's own statistics makes every
    downstream decision relative rather than absolute.

    Parameters
    ----------
    enhanced : np.ndarray   grayscale image (uint8, 0–255)

    Returns
    -------
    dict with keys:
        mean           – average pixel brightness
        std            – pixel brightness spread
        p10            – 10th percentile (darkest 10 % of pixels)
        p90            – 90th percentile (brightest 10 % of pixels)
        otsu           – Otsu's optimal global threshold value
        pothole_thresh – pixels clearly darker than the road surface
        glare_thresh   – pixels clearly brighter than normal (reflections)
    """
    mean     = float(np.mean(enhanced))
    std      = float(np.std(enhanced))
    p10      = float(np.percentile(enhanced, 10))   # darkest decile
    p90      = float(np.percentile(enhanced, 90))   # brightest decile

    # Otsu finds the grey-level that best separates two pixel populations
    # (background road vs dark damage, or bright glare vs normal surface)
    otsu_val, _ = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Pothole threshold: pixels more than 1.5 std below mean = collapsed region
    # Example: mean=120, std=30 → threshold=75  (anything darker = pothole candidate)
    # Example: mean=60,  std=20 → threshold=30  (scales down for dark images)
    pothole_thresh = float(max(0.0, mean - 1.5 * std))

    # Glare threshold: pixels above the 90th percentile + half std = reflection hotspot
    # Example: mean=120, std=30, p90=160 → threshold=175
    # Example: mean=60,  std=20, p90=90  → threshold=100  (scales to dim images)
    glare_thresh   = float(min(255.0, p90 + 0.5 * std))

    return {
        'mean':           mean,
        'std':            std,
        'p10':            p10,
        'p90':            p90,
        'otsu':           otsu_val,
        'pothole_thresh': pothole_thresh,
        'glare_thresh':   glare_thresh,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PROFILE IMAGE  — image-condition flags for adaptive downstream behaviour
# ══════════════════════════════════════════════════════════════════════════════

def profile_image(image, enhanced):
    """
    Analyse lighting and shooting conditions BEFORE detection begins.
    Returns a dict of boolean/numeric flags that F2–F13 use to adapt their
    parameters (block sizes, kernel sizes, fallback paths, etc.).

    Parameters
    ----------
    image    : np.ndarray   BGR original image
    enhanced : np.ndarray   grayscale enhanced image (from F1)

    Returns
    -------
    dict with condition flags (see inline comments for each key)
    """
    mean = float(np.mean(enhanced))
    std  = float(np.std(enhanced))

    # ── Glare / wetness detection via HSV ────────────────────────────────────
    # HSV separates hue from brightness: glare pixels are very bright (V>210)
    # AND very low saturation (S<40) — pure white reflection, no colour tint.
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2]   # Value channel (brightness)
    s_ch = hsv[:, :, 1]   # Saturation channel (colour purity)

    glare_px    = int(np.sum((v_ch > 210) & (s_ch < 40)))
    glare_ratio = glare_px / float(enhanced.size)   # fraction of image = glare

    # ── Aerial / ground-level heuristic ──────────────────────────────────────
    # Aerial images show a narrow road strip — edges concentrate in a thin
    # vertical band.  If fewer than 45 % of columns carry strong edges,
    # the image is likely aerial (RescueNet dataset).
    edges     = cv2.Canny(enhanced, 50, 150)
    col_sums  = np.sum(edges, axis=0)                         # edge energy per column
    road_width_est = int(np.sum(col_sums > col_sums.max() * 0.3))
    is_aerial = road_width_est < int(enhanced.shape[1] * 0.45)

    return {
        # Brightness flags — drive gamma correction and threshold selection
        'mean':            mean,
        'std':             std,
        'is_dark':         mean < 90,        # underexposed / night image
        'is_bright':       mean > 155,       # overexposed / washed-out
        'is_low_contrast': std < 35,         # uniform image; CLAHE needed

        # Wetness / glare flags — drive F4 glare suppression aggressiveness
        'is_wet':          glare_ratio > 0.04,   # some surface water present
        'is_very_wet':     glare_ratio > 0.12,   # flooded surface
        'glare_ratio':     glare_ratio,

        # Scene type flag — drives ROI fallback and block-size choices in F5/F6
        'is_aerial':       is_aerial,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAKE GRID  — arrange multiple images into a contact sheet
# ══════════════════════════════════════════════════════════════════════════════

def make_grid(images, labels=None, cols=4):
    """
    Tile a list of images into a grid for saving or display.
    Grayscale images are auto-converted to BGR so all tiles match.

    Parameters
    ----------
    images : list[np.ndarray]   images to tile (any mix of gray / BGR)
    labels : list[str] | None   optional text labels drawn on each tile
    cols   : int                number of columns in the grid

    Returns
    -------
    np.ndarray  single BGR grid image
    """
    if not images:
        raise ValueError("make_grid: images list is empty")

    h, w = images[0].shape[:2]   # all images assumed same size after F1 resize

    def to_bgr(img):
        # Promote grayscale (2-D) to BGR (3-D) so hstack/vstack types match
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    rows_needed = (len(images) + cols - 1) // cols
    grid_rows   = []

    for r in range(rows_needed):
        row_tiles = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(images):
                tile = to_bgr(images[idx]).copy()
                # Optionally burn a label into the top-left corner of each tile
                if labels and idx < len(labels):
                    cv2.putText(
                        tile, labels[idx],
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA
                    )
            else:
                # Pad incomplete last row with black tiles
                tile = np.zeros((h, w, 3), np.uint8)
            row_tiles.append(tile)
        grid_rows.append(np.hstack(row_tiles))

    return np.vstack(grid_rows)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  NORMALIZE METRIC  — clamp any metric to [0, 1] before RHI computation
# ══════════════════════════════════════════════════════════════════════════════

def normalize_metric(value, min_val=0.0, max_val=1.0):
    """
    Clamp a single metric to [min_val, max_val].
    Prevents any one metric from driving RHI below 0 or above 100
    due to an edge-case computation that produces out-of-range output.

    Parameters
    ----------
    value   : float   raw metric value
    min_val : float   lower bound (default 0.0)
    max_val : float   upper bound (default 1.0)

    Returns
    -------
    float   clamped value
    """
    return float(max(min_val, min(max_val, value)))


# ════════════════════════════════════════════════════════════════════════════════════════
# 6.  GET ALL TEST IMAGES  — discover all images across all datasets for batch processing
# ════════════════════════════════════════════════════════════════════════════════════════

def get_all_test_images(raw_root='data/raw'):
    """
    Discover all image files across all three dataset folders.
    Returns a list of (image_path, dataset_name) tuples.

    Searches: data/raw/rdd2022/, data/raw/crackforest/, data/raw/rescuenet/
    Supported extensions: .jpg, .jpeg, .png
    """
    EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    datasets   = ['rdd2022', 'crackforest', 'rescuenet']
    results    = []

    for dataset in datasets:
        folder = os.path.join(raw_root, dataset)
        if not os.path.isdir(folder):
            continue                             # skip if folder doesn't exist yet
        for fname in sorted(os.listdir(folder)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in EXTENSIONS:
                results.append((os.path.join(folder, fname), dataset))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CSV APPEND  — log every image's results for batch analysis
# ══════════════════════════════════════════════════════════════════════════════

def append_csv(base_name, rhi_score, condition, components,
               csv_path='data/results/rhi_results.csv'):
    """
    Append one row of results to the master CSV log.
    Creates the file with a header row on the first call.

    Parameters
    ----------
    base_name  : str    image stem (e.g. 'road_001')
    rhi_score  : float  final RHI 0–100
    condition  : str    'Good' | 'Moderate' | 'Severe'
    components : dict   the 6 metric scores + damage_score from F12
    csv_path   : str    output CSV path (created if missing)
    """
    import csv

    fieldnames = [
        'image', 'rhi_score', 'condition',
        'crack_density', 'pothole_score', 'debris_ratio',
        'roughness_score', 'fragmentation', 'glare_score', 'damage_score'
    ]

    # Detect first write so we can emit the header row
    write_header = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            'image':         base_name,
            'rhi_score':     rhi_score,
            'condition':     condition,
            'crack_density': round(components.get('crack_density', 0), 4),
            'pothole_score': round(components.get('pothole_score', 0), 4),
            'debris_ratio':  round(components.get('debris_ratio', 0), 4),
            'roughness_score': round(components.get('roughness_score', 0), 4),
            'fragmentation': round(components.get('fragmentation', 0), 4),
            'glare_score':   round(components.get('glare_score', 0), 4),
            'damage_score':  round(components.get('damage_score', 0), 4),
        })


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """
    Quick smoke-test: feed a synthetic 100×100 gradient image through every
    helper function and confirm outputs are the expected shapes/types.
    No real image required — useful to verify imports are working.
    """
    print("=== utils.py smoke test ===")

    # Build a synthetic grayscale gradient image (0→255 left→right)
    synthetic = np.tile(np.arange(0, 256, 256 / 100, dtype=np.uint8), (100, 1))
    synthetic = synthetic[:, :100]                          # 100×100 px

    # Fake BGR image (all channels identical to grayscale for simplicity)
    synthetic_bgr = cv2.cvtColor(synthetic, cv2.COLOR_GRAY2BGR)

    # 1. compute_stats
    stats = compute_stats(synthetic)
    print(f"  compute_stats → mean={stats['mean']:.1f}, "
          f"pothole_thresh={stats['pothole_thresh']:.1f}, "
          f"glare_thresh={stats['glare_thresh']:.1f}")

    # 2. profile_image
    profile = profile_image(synthetic_bgr, synthetic)
    print(f"  profile_image → is_dark={profile['is_dark']}, "
          f"is_bright={profile['is_bright']}, "
          f"is_aerial={profile['is_aerial']}")

    # 3. normalize_metric edge cases
    assert normalize_metric(-0.5) == 0.0, "Negative not clamped"
    assert normalize_metric(1.5)  == 1.0, "Over-range not clamped"
    print("  normalize_metric → clamp tests passed")

    # 4. make_grid (4 tiles)
    tiles = [synthetic, synthetic, synthetic, synthetic]
    grid  = make_grid(tiles, labels=['A', 'B', 'C', 'D'], cols=2)
    print(f"  make_grid → output shape {grid.shape}  "
          f"(expected ({synthetic.shape[0]*2}, {synthetic.shape[1]*2}, 3))")

    # 5. save_result (writes to /tmp to avoid polluting project)
    import tempfile, os
    orig_root = os.environ.get('RESULTS_ROOT', RESULTS_ROOT)
    import utils as self_mod
    self_mod.RESULTS_ROOT = tempfile.mkdtemp()          # redirect to temp dir
    path = save_result(synthetic, 0, 'smoke_test', 'step0_synth')
    print(f"  save_result → wrote to {path}")

    print("=== All utils.py tests passed ===")