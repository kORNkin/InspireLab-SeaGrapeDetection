"""
Step 7: End-to-end evaluation on the held-out test images.

Three separate numbers, deliberately not collapsed into one:

  1. Stage 1 detection -- recall and precision at IoU 0.5 against the full-image
     ground truth.
  2. Stage 2 maturity -- accuracy on the grapes stage 1 actually matched, which
     is the number that reflects the deployed pipeline (06_train_classifier.py
     already reports it on perfect ground-truth crops).
  3. Per-image counts -- what a grower reads off the output.

Read precision with care. The annotators labelled a subset of the resolvable
spheres, so a detector that correctly finds an unlabelled grape is punished for
it. Precision here is a lower bound and mostly measures annotation density.
Recall against labelled instances is the trustworthy figure.

Also benchmarks --downscale 1 vs 2 so the speed/recall trade is a measurement
rather than a guess.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import TwoStageDetector  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_IMAGES = BASE_DIR / "build" / "detect" / "test" / "images"
TEST_LABELS = BASE_DIR / "build" / "detect" / "test" / "labels"
OUT = BASE_DIR / "build" / "eval"

CLASS_NAMES = ["Darkening", "Harvestable", "Whitening"]


def load_gt(label_path, W, H):
    boxes, classes = [], []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        c, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H
        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        classes.append(int(c))
    return np.array(boxes).reshape(-1, 4), np.array(classes, int)


def iou_matrix(a, b):
    """Pairwise IoU between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / (area_a + area_b - inter + 1e-9)


def greedy_match(pred, gt, scores, thr=0.5):
    """Highest-score-first matching. Returns pred->gt index, -1 for unmatched."""
    assign = np.full(len(pred), -1, int)
    if len(pred) == 0 or len(gt) == 0:
        return assign
    ious = iou_matrix(pred, gt)
    taken = np.zeros(len(gt), bool)
    for pi in np.argsort(-scores):
        row = ious[pi].copy()
        row[taken] = -1
        gi = int(row.argmax())
        if row[gi] >= thr:
            assign[pi] = gi
            taken[gi] = True
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--downscale", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--save-vis", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    det = TwoStageDetector(conf=args.conf)
    images = sorted(TEST_IMAGES.glob("*"))
    print(f"Test images: {len(images)}\n")

    report = {}
    for ds in args.downscale:
        print(f"{'=' * 78}\n=== downscale {ds}x " + ("(full resolution)" if ds == 1 else
              f"({4640 // ds}x{3480 // ds})") + f"\n{'=' * 78}")

        tp = fp = fn = 0
        cls_right = cls_total = 0
        cls_conf = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES) + 1), int)  # +1 = Uncertain
        per_image, times = [], []

        for img_path in images:
            img = cv2.imread(str(img_path))
            H, W = img.shape[:2]
            gt_boxes, gt_cls = load_gt(TEST_LABELS / (img_path.stem + ".txt"), W, H)
            stem = img_path.name.split("_")[0]

            t0 = time.perf_counter()
            boxes, scores = det.detect(img, downscale=ds)
            pred_cls, pred_prob = det.classify(img, boxes)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

            pb, sc = boxes.numpy(), scores.numpy()
            assign = greedy_match(pb, gt_boxes, sc, args.iou)

            matched = int((assign >= 0).sum())
            tp += matched
            fp += len(pb) - matched
            fn += len(gt_boxes) - matched

            # Maturity accuracy, scored only on detections that hit a real grape.
            for pi, gi in enumerate(assign):
                if gi < 0:
                    continue
                truth = gt_cls[gi]
                cls_total += 1
                if pred_prob[pi] < det.uncertain_below:
                    cls_conf[truth, len(CLASS_NAMES)] += 1
                else:
                    cls_conf[truth, pred_cls[pi]] += 1
                    if pred_cls[pi] == truth:
                        cls_right += 1

            gt_counts = np.bincount(gt_cls, minlength=len(CLASS_NAMES))
            pr_counts = np.bincount(
                pred_cls[pred_prob >= det.uncertain_below], minlength=len(CLASS_NAMES)
            )
            per_image.append({
                "image": stem, "gt_total": len(gt_boxes), "pred_total": len(pb),
                "recall": matched / len(gt_boxes) if len(gt_boxes) else 0.0,
                "gt_counts": gt_counts.tolist(), "pred_counts": pr_counts.tolist(),
                "ms": elapsed,
            })

            if args.save_vis and ds == 1:
                dets = [{"box": b.tolist(),
                         "class": CLASS_NAMES[c] if p >= det.uncertain_below else "Uncertain",
                         "classifier_conf": float(p), "detector_conf": float(s)}
                        for b, c, p, s in zip(pb, pred_cls, pred_prob, sc)]
                cov, mask = det.slime_coverage(img)
                cv2.imwrite(str(OUT / f"{stem}_pred.jpg"), det.draw(img, dets, mask, cov))

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0

        print(f"\n--- Stage 1 detection @ IoU {args.iou} ---")
        print(f"  Recall    : {recall:.4f}  ({tp}/{tp + fn} labelled grapes found)   <- trustworthy")
        print(f"  Precision : {precision:.4f}  ({fp} unmatched)   <- lower bound, see note")
        print(f"  Note: unlabelled-but-real grapes are counted as false positives here.")

        print(f"\n--- Stage 2 maturity, on matched detections ---")
        if cls_total:
            print(f"{'':<14}" + "".join(f"{n[:11]:>12}" for n in CLASS_NAMES)
                  + f"{'Uncertain':>12}{'recall':>9}")
            per_cls_recall = []
            for i, name in enumerate(CLASS_NAMES):
                tot = cls_conf[i].sum()
                r = cls_conf[i, i] / tot if tot else 0.0
                per_cls_recall.append(r)
                print(f"{name:<14}" + "".join(f"{v:>12}" for v in cls_conf[i]) + f"{r:>9.3f}")
            print(f"  Accuracy     : {cls_right / cls_total:.4f}")
            print(f"  Macro recall : {np.mean(per_cls_recall):.4f}")
        else:
            per_cls_recall = [0.0] * len(CLASS_NAMES)
            print("  no matched detections")

        print(f"\n--- Per-image counts ---")
        print(f"  {'img':<6}{'GT':>6}{'pred':>7}{'recall':>9}{'ms':>9}")
        for r in per_image:
            print(f"  {r['image']:<6}{r['gt_total']:>6}{r['pred_total']:>7}"
                  f"{r['recall']:>9.3f}{r['ms']:>9.0f}")
        print(f"\n  Mean latency: {np.mean(times):.0f} ms/image ({1000 / np.mean(times):.2f} img/s)")

        report[f"downscale_{ds}"] = {
            "recall": recall, "precision": precision, "tp": tp, "fp": fp, "fn": fn,
            "maturity_accuracy": cls_right / cls_total if cls_total else 0.0,
            "maturity_macro_recall": float(np.mean(per_cls_recall)),
            "maturity_confusion": cls_conf.tolist(),
            "mean_ms": float(np.mean(times)),
            "per_image": per_image,
        }
        print()

    if len(args.downscale) > 1:
        print(f"{'=' * 78}\n=== Speed / recall trade\n{'=' * 78}")
        print(f"  {'downscale':<12}{'recall':>10}{'ms/image':>12}{'speedup':>10}")
        base = report[f"downscale_{args.downscale[0]}"]["mean_ms"]
        for ds in args.downscale:
            r = report[f"downscale_{ds}"]
            print(f"  {ds}x{'':<10}{r['recall']:>10.4f}{r['mean_ms']:>12.0f}"
                  f"{base / r['mean_ms']:>9.2f}x")

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
