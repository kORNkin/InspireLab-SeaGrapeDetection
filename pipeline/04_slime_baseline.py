"""
Step 4: RedSlime coverage without training a model.

RedSlime is a pink encrusting mat over the substrate, not a countable object and
not a grape maturity stage -- 0 of 29 boxes in image 029 and only 7 of 73 in 030
fall inside a polygon. What the grower wants from it is a coverage percentage,
so this estimates a binary mask and reports the area fraction.

With 11 polygons across 2 images there is nothing to train on, but the target is
a flat saturated pink against green/brown substrate, which a colour threshold
handles. This grid-searches an HSV box plus morphology and reports IoU.

Honesty about the number: n=2. Fitting and scoring on both images would only
measure memorisation, so the search is leave-one-image-out -- thresholds are
tuned on one image and scored on the other. The reported IoU is the held-out
one. Treat it as a smoke test, not a benchmark.

Runs at 1/4 resolution (1160x870). Slime patches are 0.09-5.8 MP; they survive
downscaling intact, and it makes the search cheap.
"""

import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
SRC = BASE_DIR / "build" / "slime"
OUT = BASE_DIR / "build" / "slime_baseline"

DOWNSCALE = 4
MIN_BLOB_FRAC = 0.002  # of image area; kills red netting speckle and algae fronds


def load_pairs():
    """Yield (stem, bgr_small, gt_mask_small) for every polygon-labelled image."""
    pairs = []
    for split in ["train", "valid", "test"]:
        img_dir = SRC / split / "images"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            small = cv2.resize(img, (W // DOWNSCALE, H // DOWNSCALE), interpolation=cv2.INTER_AREA)
            sh, sw = small.shape[:2]

            mask = np.zeros((sh, sw), np.uint8)
            label_path = SRC / split / "labels" / (img_path.stem + ".txt")
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                coords = [float(v) for v in line.split()[1:]]
                pts = (np.array(coords).reshape(-1, 2) * [sw, sh]).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
            pairs.append((img_path.name.split("_")[0], small, mask))
    return pairs


def segment(bgr, params):
    """HSV box threshold -> morphological close/open -> min-area blob filter."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    lo = np.array([params["h_lo"], params["s_lo"], params["v_lo"]], np.uint8)
    hi = np.array([params["h_hi"], params["s_hi"], params["v_hi"]], np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    k = np.ones((params["k"], params["k"]), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # Slime forms large contiguous mats; anything small is netting or algae.
    min_area = MIN_BLOB_FRAC * mask.size
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return (keep > 0).astype(np.uint8)


def iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union else 1.0


def search(pairs):
    """Grid-search the HSV box, scoring mean IoU over the given images."""
    grid = product(
        [130, 140, 150, 160],       # h_lo
        [175, 179],                 # h_hi
        [20, 30, 40, 50],           # s_lo
        [255],                      # s_hi
        [60, 90, 120],              # v_lo
        [255],                      # v_hi
        [5, 9],                     # morphology kernel
    )
    best, best_score = None, -1.0
    for h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, k in grid:
        if h_lo >= h_hi:
            continue
        p = dict(h_lo=h_lo, h_hi=h_hi, s_lo=s_lo, s_hi=s_hi, v_lo=v_lo, v_hi=v_hi, k=k)
        score = float(np.mean([iou(segment(img, p), gt) for _, img, gt in pairs]))
        if score > best_score:
            best, best_score = p, score
    return best, best_score


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    print(f"Polygon-labelled images: {len(pairs)} ({', '.join(s for s, _, _ in pairs)})\n")
    if len(pairs) < 2:
        print("Need at least 2 labelled images for leave-one-out. Aborting.")
        return

    print("=== Leave-one-image-out (the honest number) ===")
    held_scores = []
    for i, (stem, img, gt) in enumerate(pairs):
        train_pairs = [p for j, p in enumerate(pairs) if j != i]
        params, fit_score = search(train_pairs)
        held = iou(segment(img, params), gt)
        held_scores.append(held)
        print(f"  tuned on {[p[0] for p in train_pairs]} (IoU {fit_score:.3f}) "
              f"-> held-out {stem}: IoU {held:.3f}")
    print(f"  mean held-out IoU: {np.mean(held_scores):.3f}\n")

    print("=== Final params, fitted on all available images ===")
    params, fit_score = search(pairs)
    print(f"  {params}")
    print(f"  fitted IoU (memorisation, not performance): {fit_score:.3f}\n")

    # Per-image coverage and a visual for eyeballing.
    print("=== Coverage estimate vs ground truth ===")
    for stem, img, gt in pairs:
        pred = segment(img, params)
        print(f"  {stem}: predicted {100 * pred.mean():5.1f}%  |  "
              f"actual {100 * gt.mean():5.1f}%  |  IoU {iou(pred, gt):.3f}")

        vis = img.copy()
        vis[gt > 0] = (0.6 * vis[gt > 0] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
        overlay = img.copy()
        overlay[pred > 0] = (0.6 * overlay[pred > 0] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(str(OUT / f"{stem}_gt_vs_pred.jpg"), np.hstack([vis, overlay]))

    (OUT / "params.json").write_text(json.dumps(
        {"hsv": params, "downscale": DOWNSCALE, "min_blob_frac": MIN_BLOB_FRAC,
         "mean_holdout_iou": float(np.mean(held_scores)), "n_labelled_images": len(pairs)},
        indent=2,
    ))
    print(f"\nWrote {OUT}/params.json and side-by-side visuals (left=GT green, right=pred red)")


if __name__ == "__main__":
    main()
