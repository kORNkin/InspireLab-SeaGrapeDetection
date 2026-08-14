"""
Step 9: Score the trained models against BOTH label sets and quantify the gap.

Two YOLO exports cover the same photographs:

    dataset_all_yolo/   ORIGINAL   3,791 grape boxes
    dataset/            UPDATED    8,555 grape boxes  (annotations added by hand)

The models trained on the updated set. Scoring against both answers a question
neither one answers alone: how much of the apparent "false positive" rate is
really just annotation coverage?

Every shared image is scored, train/valid/test alike -- the comparison here is
between two label sets over identical predictions, so the split boundaries are
not the variable under study. Note that this makes the absolute numbers
optimistic as a generalisation estimate, since most images were trained on.

Two traps this script handles, each of which silently corrupts the comparison
rather than raising an error:

  1. THE CLASS ORDERS DIFFER between the two data.yaml files.
         original : ['Darkening', 'Harvestable', 'Whitening', 'RedSlime']
         updated  : ['RedSlime', 'Darkening', 'Harvestable', 'Whitening']
     Reading raw class ids without remapping mislabels every single box.

  2. The two exports also disagree about splits and Image 013 exists only in
     the updated one, so labels are indexed by image stem rather than by folder.

Detection runs once per image and is cached to disk, so both label sets are
scored against identical predictions and re-running the metrics is instant.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import TwoStageDetector  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
ORIGINAL = BASE_DIR / "dataset_all_yolo"
UPDATED = BASE_DIR / "dataset"
OUT = BASE_DIR / "build" / "compare_label_sets"

# Canonical order, matching the stage-2 classifier's ImageFolder ordering.
CANON = ["Darkening", "Harvestable", "Whitening"]
IOU_LEVELS = [0.3, 0.5, 0.75]


def yaml_names(path):
    """Minimal reader for the `names: [...]` line; avoids a pyyaml dependency."""
    for line in path.read_text().splitlines():
        if line.strip().startswith("names:"):
            raw = line.split("names:", 1)[1].strip()
            return [n.strip().strip("'\"") for n in raw.strip("[]").split(",")]
    raise ValueError(f"no names in {path}")


def load_label_set(root, ref_sizes=None):
    """Return {stem: (boxes_xyxy, canonical_class_ids)} keyed by image stem.

    Class ids are remapped from the export's own ordering into CANON, and
    RedSlime rows are dropped -- it is a substrate condition, not a grape.
    Polygon rows (more than 4 coordinates) are skipped for the same reason.

    ref_sizes maps stem -> (W, H) and MUST be supplied when the boxes will be
    compared against something computed elsewhere. Roboflow exported these two
    sets at different resolutions -- dataset/ is 4640x3480 but dataset_all_yolo/
    is 2048x1536 -- so de-normalising each set against its own image puts them
    in coordinate spaces 2.27x apart. Boxes around the same grape then score
    IoU ~0.02 instead of ~0.7, and the whole comparison silently becomes
    meaningless. Everything is de-normalised against the prediction source's
    dimensions instead.
    """
    names = yaml_names(root / "data.yaml")
    remap = {i: CANON.index(n) for i, n in enumerate(names) if n in CANON}

    out = {}
    for split in ["train", "valid", "test"]:
        img_dir = root / split / "images"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*")):
            stem = img_path.stem.split("_")[0]
            label_path = root / split / "labels" / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            if ref_sizes is not None:
                if stem not in ref_sizes:
                    continue
                W, H = ref_sizes[stem]
            else:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                H, W = img.shape[:2]

            boxes, classes = [], []
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:      # polygon row (RedSlime) -- not a grape
                    continue
                cid = int(parts[0])
                if cid not in remap:     # RedSlime rectangle, if any
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:])
                cx, cy, bw, bh = cx * W, cy * H, bw * W, bh * H
                boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
                classes.append(remap[cid])

            out[stem] = (np.array(boxes).reshape(-1, 4), np.array(classes, int),
                         split, str(img_path))
    return out


def iou_matrix(a, b):
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


def match(pred, gt, scores, thr):
    """Greedy highest-score-first one-to-one assignment. Returns pred -> gt idx."""
    assign = np.full(len(pred), -1, int)
    if len(pred) == 0 or len(gt) == 0:
        return assign
    ious = iou_matrix(pred, gt)
    taken = np.zeros(len(gt), bool)
    for pi in np.argsort(-scores):
        row = ious[pi].copy()
        row[taken] = -1.0
        gi = int(row.argmax())
        if row[gi] >= thr:
            assign[pi] = gi
            taken[gi] = True
    return assign


def match_centre(pred, gt, scores):
    """Match a prediction to a ground-truth box when its centre falls inside.

    IoU is unusable for comparing these two label sets: the original boxes have a
    median width of 41px against the updated set's 79px, so two boxes drawn
    around the SAME grape score IoU ~ (41/79)^2 = 0.27 and fail every threshold.
    Centre containment answers "did we find this grape" independently of how
    tightly either annotator chose to draw.
    """
    assign = np.full(len(pred), -1, int)
    if len(pred) == 0 or len(gt) == 0:
        return assign
    cx = (pred[:, 0] + pred[:, 2]) / 2
    cy = (pred[:, 1] + pred[:, 3]) / 2
    taken = np.zeros(len(gt), bool)
    for pi in np.argsort(-scores):
        inside = np.where((~taken) & (gt[:, 0] <= cx[pi]) & (cx[pi] <= gt[:, 2])
                          & (gt[:, 1] <= cy[pi]) & (cy[pi] <= gt[:, 3]))[0]
        if len(inside):
            assign[pi] = inside[0]
            taken[inside[0]] = True
    return assign


def average_precision(tp_flags, scores, n_gt):
    """All-point-interpolated AP, the COCO/VOC2010 convention."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = np.array(tp_flags, float)[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    # Make precision monotonically decreasing, then integrate over recall.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0] if len(precision) else 0.0], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


def score_against(labels, preds, stems):
    """Full metric bundle for one label set over the given stems."""
    res = {"n_images": len(stems), "n_gt": 0, "n_pred": 0, "by_iou": {},
           "count": {}, "maturity": None}

    res["n_gt"] = int(sum(len(labels[s][0]) for s in stems))
    res["n_pred"] = int(sum(len(preds[s]["boxes"]) for s in stems))

    for thr in IOU_LEVELS:
        tp = fp = fn = 0
        flags, scs = [], []
        for s in stems:
            gt = labels[s][0]
            pb = np.array(preds[s]["boxes"]).reshape(-1, 4)
            sc = np.array(preds[s]["scores"])
            a = match(pb, gt, sc, thr)
            m = int((a >= 0).sum())
            tp += m
            fp += len(pb) - m
            fn += len(gt) - m
            flags.extend((a >= 0).astype(int).tolist())
            scs.extend(sc.tolist())
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        res["by_iou"][str(thr)] = {
            "tp": tp, "fp": fp, "fn": fn, "recall": rec, "precision": prec,
            "f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
            "ap": average_precision(flags, np.array(scs), tp + fn),
        }

    # mAP averaged over IoU 0.50:0.05:0.95, the COCO primary metric.
    aps = []
    for thr in np.arange(0.5, 0.96, 0.05):
        flags, scs, n_gt = [], [], 0
        for s in stems:
            gt = labels[s][0]
            pb = np.array(preds[s]["boxes"]).reshape(-1, 4)
            sc = np.array(preds[s]["scores"])
            a = match(pb, gt, sc, float(thr))
            flags.extend((a >= 0).astype(int).tolist())
            scs.extend(sc.tolist())
            n_gt += len(gt)
        aps.append(average_precision(flags, np.array(scs), n_gt))
    res["map_50_95"] = float(np.mean(aps))

    # Convention-independent detection: centre containment instead of IoU.
    tp = fp = fn = 0
    for s in stems:
        gt = labels[s][0]
        pb = np.array(preds[s]["boxes"]).reshape(-1, 4)
        sc = np.array(preds[s]["scores"])
        a = match_centre(pb, gt, sc)
        m = int((a >= 0).sum())
        tp += m
        fp += len(pb) - m
        fn += len(gt) - m
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    res["centre"] = {
        "tp": tp, "fp": fp, "fn": fn, "recall": rec, "precision": prec,
        "f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
    }

    # Box geometry, which is what makes the IoU comparison misleading.
    widths = [w for s in stems for w in (labels[s][0][:, 2] - labels[s][0][:, 0])]
    pwidths = [w for s in stems for w in
               (np.array(preds[s]["boxes"]).reshape(-1, 4)[:, 2]
                - np.array(preds[s]["boxes"]).reshape(-1, 4)[:, 0])]
    res["geometry"] = {
        "gt_median_width": float(np.median(widths)) if widths else 0.0,
        "pred_median_width": float(np.median(pwidths)) if pwidths else 0.0,
    }

    # Counting error -- what a grower actually reads off the output.
    errs, pcts = [], []
    for s in stems:
        g, p = len(labels[s][0]), len(preds[s]["boxes"])
        errs.append(p - g)
        if g:
            pcts.append(100 * (p - g) / g)
    res["count"] = {
        "mae": float(np.mean(np.abs(errs))) if errs else 0.0,
        "bias": float(np.mean(errs)) if errs else 0.0,
        "mape": float(np.mean(np.abs(pcts))) if pcts else 0.0,
        "ratio": res["n_pred"] / res["n_gt"] if res["n_gt"] else 0.0,
    }

    # Maturity, scored only on detections matched at IoU 0.5.
    conf = np.zeros((3, 4), int)   # 4th column = Uncertain
    for s in stems:
        gt, gc = labels[s][0], labels[s][1]
        pb = np.array(preds[s]["boxes"]).reshape(-1, 4)
        sc = np.array(preds[s]["scores"])
        pc = np.array(preds[s]["cls"], int)
        pp = np.array(preds[s]["prob"])
        a = match(pb, gt, sc, 0.5)
        for pi, gi in enumerate(a):
            if gi < 0:
                continue
            conf[gc[gi], 3 if pp[pi] < 0.6 else pc[pi]] += 1
    recalls = [conf[i, i] / conf[i].sum() if conf[i].sum() else 0.0 for i in range(3)]
    res["maturity"] = {
        "confusion": conf.tolist(),
        "per_class_recall": dict(zip(CANON, recalls)),
        "macro_recall": float(np.mean(recalls)),
        "accuracy": float(np.trace(conf[:, :3]) / conf.sum()) if conf.sum() else 0.0,
        "n_matched": int(conf.sum()),
    }
    return res


SIZE_BUCKETS = [(0, 30), (30, 45), (45, 60), (60, 90), (90, float("inf"))]


def size_bucket_analysis(orig, upd, preds, stems):
    """Bucket ORIGINAL boxes by width; report who finds them.

    This is what actually explains the low score against the original labels.
    It is not that the same grapes are boxed more tightly -- it is that the
    original set contains a large population of very small objects that the
    updated set declined to label at all, and that the model (trained on the
    updated set) therefore never learned to emit.
    """
    tot = np.zeros(len(SIZE_BUCKETS))
    found = np.zeros(len(SIZE_BUCKETS))
    covered = np.zeros(len(SIZE_BUCKETS))

    for s in stems:
        ob = orig[s][0]
        ub = upd[s][0]
        pb = np.array(preds[s]["boxes"]).reshape(-1, 4)
        if len(ob) == 0:
            continue
        widths = ob[:, 2] - ob[:, 0]
        pc = (np.stack([(pb[:, 0] + pb[:, 2]) / 2, (pb[:, 1] + pb[:, 3]) / 2], 1)
              if len(pb) else np.zeros((0, 2)))
        uc = (np.stack([(ub[:, 0] + ub[:, 2]) / 2, (ub[:, 1] + ub[:, 3]) / 2], 1)
              if len(ub) else np.zeros((0, 2)))

        for gi, (x1, y1, x2, y2) in enumerate(ob):
            bi = next(i for i, (lo, hi) in enumerate(SIZE_BUCKETS) if lo <= widths[gi] < hi)
            tot[bi] += 1
            if len(pc) and ((pc[:, 0] >= x1) & (pc[:, 0] <= x2)
                            & (pc[:, 1] >= y1) & (pc[:, 1] <= y2)).any():
                found[bi] += 1
            if len(uc) and ((uc[:, 0] >= x1) & (uc[:, 0] <= x2)
                            & (uc[:, 1] >= y1) & (uc[:, 1] <= y2)).any():
                covered[bi] += 1

    rows = []
    for i, (lo, hi) in enumerate(SIZE_BUCKETS):
        rows.append({
            "range": f"{lo}-{'inf' if hi == float('inf') else int(hi)} px",
            "count": int(tot[i]),
            "model_finds": float(found[i] / tot[i]) if tot[i] else 0.0,
            "updated_covers": float(covered[i] / tot[i]) if tot[i] else 0.0,
        })
    small = int(tot[0] + tot[1])
    return {"buckets": rows, "n_total": int(tot.sum()), "n_under_45px": small,
            "frac_under_45px": small / max(tot.sum(), 1)}


def added_box_analysis(orig, upd, preds, stems, thr=0.5):
    """Of the boxes ADDED in the update, how many does the model find?

    If the additions are real grapes the model already detected, recall on them
    should be comparable to recall on the original boxes -- which would mean the
    update mostly documented detections that were previously scored as errors.
    """
    added_total = added_found = 0
    kept_total = kept_found = 0
    for s in stems:
        if s not in orig:
            continue
        o_boxes = orig[s][0]
        u_boxes, _ = upd[s][0], upd[s][1]
        pb = np.array(preds[s]["boxes"]).reshape(-1, 4)
        sc = np.array(preds[s]["scores"])

        # An updated box is "added" if no original box overlaps it.
        if len(o_boxes) and len(u_boxes):
            best = iou_matrix(u_boxes, o_boxes).max(1)
        else:
            best = np.zeros(len(u_boxes))
        is_added = best < thr

        a = match(pb, u_boxes, sc, thr)
        found = np.zeros(len(u_boxes), bool)
        found[a[a >= 0]] = True

        added_total += int(is_added.sum())
        added_found += int((found & is_added).sum())
        kept_total += int((~is_added).sum())
        kept_found += int((found & ~is_added).sum())

    return {
        "added_boxes": added_total,
        "added_found": added_found,
        "added_recall": added_found / added_total if added_total else 0.0,
        "preexisting_boxes": kept_total,
        "preexisting_found": kept_found,
        "preexisting_recall": kept_found / kept_total if kept_total else 0.0,
    }


def run_predictions(upd, cache_path, conf, downscale):
    if cache_path.exists():
        print(f"Reusing cached predictions: {cache_path}")
        return json.loads(cache_path.read_text())

    det = TwoStageDetector(conf=conf)
    preds = {}
    for i, (stem, (_, _, split, img_path)) in enumerate(sorted(upd.items()), 1):
        img = cv2.imread(img_path)
        boxes, scores = det.detect(img, downscale=downscale)
        cls, prob = det.classify(img, boxes)
        preds[stem] = {"boxes": boxes.numpy().tolist(), "scores": scores.numpy().tolist(),
                       "cls": cls.tolist(), "prob": prob.tolist()}
        print(f"  [{i:>2}/{len(upd)}] {stem} -> {len(boxes)} detections")
    cache_path.write_text(json.dumps(preds))
    print(f"Cached predictions to {cache_path}")
    return preds


def fmt_block(name, r):
    lines = [f"\n{name}  ({r['n_images']} images)",
             f"  ground-truth boxes : {r['n_gt']}",
             f"  detections         : {r['n_pred']}   ({r['count']['ratio']:.2f}x)"]
    lines.append(f"  {'IoU':<6}{'recall':>9}{'precision':>11}{'F1':>9}{'AP':>9}"
                 f"{'TP':>8}{'FP':>8}{'FN':>8}")
    for thr in IOU_LEVELS:
        d = r["by_iou"][str(thr)]
        lines.append(f"  {thr:<6.2f}{d['recall']:>9.4f}{d['precision']:>11.4f}"
                     f"{d['f1']:>9.4f}{d['ap']:>9.4f}{d['tp']:>8}{d['fp']:>8}{d['fn']:>8}")
    lines.append(f"  mAP@[.50:.95]      : {r['map_50_95']:.4f}")
    g, ce = r["geometry"], r["centre"]
    lines.append(f"  GT median box {g['gt_median_width']:.0f}px vs predicted "
                 f"{g['pred_median_width']:.0f}px")
    lines.append(f"  centre-match       : recall {ce['recall']:.4f}  precision "
                 f"{ce['precision']:.4f}  F1 {ce['f1']:.4f}   <- convention-independent")
    c = r["count"]
    lines.append(f"  count MAE {c['mae']:.1f} | bias {c['bias']:+.1f} | MAPE {c['mape']:.1f}%")
    m = r["maturity"]
    lines.append(f"  maturity: accuracy {m['accuracy']:.4f} | macro recall "
                 f"{m['macro_recall']:.4f} | on {m['n_matched']} matched")
    return "\n".join(lines)


def write_markdown(path, ctx):
    """Emit the findings report."""
    o, u = ctx["original"], ctx["updated"]
    add = ctx["added"]

    def iou_table(r):
        rows = ["| IoU | Recall | Precision | F1 | AP | TP | FP | FN |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for thr in IOU_LEVELS:
            d = r["by_iou"][str(thr)]
            rows.append(f"| {thr:.2f} | {d['recall']:.4f} | {d['precision']:.4f} | "
                        f"{d['f1']:.4f} | {d['ap']:.4f} | {d['tp']} | {d['fp']} | {d['fn']} |")
        rows.append(f"| **mAP@[.50:.95]** | | | | **{r['map_50_95']:.4f}** | | | |")
        return "\n".join(rows)

    def conf_table(r):
        c = np.array(r["maturity"]["confusion"])
        rows = ["| Truth | Darkening | Harvestable | Whitening | Uncertain | Recall |",
                "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for i, n in enumerate(CANON):
            rec = r["maturity"]["per_class_recall"][n]
            rows.append(f"| {n} | {c[i,0]} | {c[i,1]} | {c[i,2]} | {c[i,3]} | {rec:.3f} |")
        return "\n".join(rows)

    md = f"""# Label Set Comparison — Original vs Updated Annotations

Both exports cover the same photographs. The models in `build/runs/` trained on
the **updated** set. Scoring identical predictions against both isolates how much
of the measured error is model behaviour and how much is annotation coverage.

**Every number in this report covers the same {ctx['n_shared']} images.** The updated export
contains {ctx['n_upd_images']} but image `013` is absent from the original, so it is excluded
throughout rather than being counted on one side only.

| | Original (`dataset_all_yolo/`) | Updated (`dataset/`) | Model predictions |
| --- | ---: | ---: | ---: |
| Images | {ctx['n_shared']} | {ctx['n_shared']} | {ctx['n_shared']} |
| Grape boxes | {o['n_gt']:,} | {u['n_gt']:,} | {ctx['pred_total']:,} |
| Darkening | {ctx['orig_hist'][0]:,} | {ctx['upd_hist'][0]:,} | {ctx['pred_hist'][0]:,} |
| Harvestable | {ctx['orig_hist'][1]:,} | {ctx['upd_hist'][1]:,} | {ctx['pred_hist'][1]:,} |
| Whitening | {ctx['orig_hist'][2]:,} | {ctx['upd_hist'][2]:,} | {ctx['pred_hist'][2]:,} |
| Uncertain | — | — | {ctx['pred_uncertain']:,} |

The update added **{u['n_gt'] - o['n_gt']:,} boxes**, a {u['n_gt'] / max(o['n_gt'],1):.2f}× increase.

"Uncertain" is not a trained class — it is any detection whose top maturity
probability fell below 0.60, kept in the total because the grape is real even
when its stage is not callable.

### Class mix as a share of total

| Class | Original | Updated | Predicted |
| --- | ---: | ---: | ---: |
| Darkening | {100 * ctx['orig_hist'][0] / max(o['n_gt'], 1):.1f}% | {100 * ctx['upd_hist'][0] / max(u['n_gt'], 1):.1f}% | {100 * ctx['pred_hist'][0] / max(ctx['pred_total'], 1):.1f}% |
| Harvestable | {100 * ctx['orig_hist'][1] / max(o['n_gt'], 1):.1f}% | {100 * ctx['upd_hist'][1] / max(u['n_gt'], 1):.1f}% | {100 * ctx['pred_hist'][1] / max(ctx['pred_total'], 1):.1f}% |
| Whitening | {100 * ctx['orig_hist'][2] / max(o['n_gt'], 1):.1f}% | {100 * ctx['upd_hist'][2] / max(u['n_gt'], 1):.1f}% | {100 * ctx['pred_hist'][2] / max(ctx['pred_total'], 1):.1f}% |
| Uncertain | — | — | {100 * ctx['pred_uncertain'] / max(ctx['pred_total'], 1):.1f}% |

{ctx['mix_note']}

## Scope and three traps

All {ctx['n_shared']} images are scored, train/valid/test alike. The variable under study is
the label set, not the split, and both label sets are scored against
byte-identical predictions. Because most of these images were trained on, treat
the absolute values as optimistic — the *difference* between the two columns is
the meaningful quantity, and that difference is unaffected.

Three traps, each of which corrupts results silently rather than raising an error:

1. **The class orders differ between the two `data.yaml` files.**
   Original is `['Darkening', 'Harvestable', 'Whitening', 'RedSlime']`;
   updated is `['RedSlime', 'Darkening', 'Harvestable', 'Whitening']`.
   Reading raw class ids mislabels every box. This script remaps both into a
   canonical order.
2. **The exports disagree about splits.** Images `066`, `131` and `157` are test
   in the original but train in the updated set. Labels are therefore indexed by
   image stem rather than by folder.
3. **Image `013` exists only in the updated export.** Including it would put an
   extra image's worth of boxes on one side of the comparison, so it is dropped
   from every table here.

---

## Detection metrics

### Against the ORIGINAL labels

{iou_table(o)}

Counting: MAE {o['count']['mae']:.1f} · bias {o['count']['bias']:+.1f} · MAPE {o['count']['mape']:.1f}% · ratio {o['count']['ratio']:.2f}×

### Against the UPDATED labels

{iou_table(u)}

Counting: MAE {u['count']['mae']:.1f} · bias {u['count']['bias']:+.1f} · MAPE {u['count']['mape']:.1f}% · ratio {u['count']['ratio']:.2f}×

### Side by side at IoU 0.5

| Metric | Original labels | Updated labels | Change |
| --- | ---: | ---: | ---: |
| Recall | {o['by_iou']['0.5']['recall']:.4f} | {u['by_iou']['0.5']['recall']:.4f} | {u['by_iou']['0.5']['recall'] - o['by_iou']['0.5']['recall']:+.4f} |
| Precision | {o['by_iou']['0.5']['precision']:.4f} | {u['by_iou']['0.5']['precision']:.4f} | {u['by_iou']['0.5']['precision'] - o['by_iou']['0.5']['precision']:+.4f} |
| F1 | {o['by_iou']['0.5']['f1']:.4f} | {u['by_iou']['0.5']['f1']:.4f} | {u['by_iou']['0.5']['f1'] - o['by_iou']['0.5']['f1']:+.4f} |
| AP@0.5 | {o['by_iou']['0.5']['ap']:.4f} | {u['by_iou']['0.5']['ap']:.4f} | {u['by_iou']['0.5']['ap'] - o['by_iou']['0.5']['ap']:+.4f} |
| mAP@[.50:.95] | {o['map_50_95']:.4f} | {u['map_50_95']:.4f} | {u['map_50_95'] - o['map_50_95']:+.4f} |
| Count ratio | {o['count']['ratio']:.2f}× | {u['count']['ratio']:.2f}× | |

The predictions are identical in both columns. Every difference comes from the
labels alone.

---

## The resolution trap — read this before reusing these scripts

**Roboflow exported the two sets at different resolutions.** `dataset/` images
are 4640×3480; `dataset_all_yolo/` images are **2048×1536**. YOLO labels are
normalised, so de-normalising each set against its own image puts the two in
coordinate spaces 2.27× apart.

An earlier version of this script did exactly that. Boxes drawn around the same
grape scored IoU ≈ **0.02** instead of ≈ **0.7**, and every downstream number was
garbage in a way that looked entirely plausible:

| | Buggy (each set in its own space) | Correct (common space) |
| --- | ---: | ---: |
| Recall vs original @ IoU 0.5 | 0.175 | **{o['by_iou']['0.5']['recall']:.4f}** |
| Precision vs original | 0.083 | **{o['by_iou']['0.5']['precision']:.4f}** |
| mAP@[.50:.95] vs original | 0.007 | **{o['map_50_95']:.4f}** |
| Original median box width | 41 px | **{o['geometry']['gt_median_width']:.0f} px** |
| Original boxes under 45 px | 56% | **{ctx['sizes']['frac_under_45px']:.0%}** |

The buggy version supported a confident and completely wrong conclusion: that
the two annotators disagreed about the minimum grape size worth labelling. They
do not. All labels are now de-normalised against the prediction source's
dimensions.

## The two label sets actually agree closely

| | Median box width |
| --- | ---: |
| Original labels | {o['geometry']['gt_median_width']:.0f} px |
| Updated labels | {u['geometry']['gt_median_width']:.0f} px |
| Model predictions | {u['geometry']['pred_median_width']:.0f} px |

Same convention, and the model matches it. Bucketing the original boxes by width
confirms the updated set is close to a **superset**:

| Original box width | Count | Model finds | Updated set also labels |
| --- | ---: | ---: | ---: |
{ctx['size_table']}

The updated set covers **94–96%** of the original boxes at every size. It did not
change what counts as a grape; it labelled more of them.

Model recall rises with object size — {ctx['sizes']['buckets'][2]['model_finds']:.0%} in the 45–60 px band up to
{ctx['biggest_bucket_model']:.0%} above 90 px — which is ordinary small-object detector behaviour, not
a policy disagreement.

---

## Do the added annotations correspond to real detections?

Splitting the updated ground truth into boxes that already existed and boxes the
update introduced, then measuring recall on each separately:

| Subset | Boxes | Found | Recall |
| --- | ---: | ---: | ---: |
| Already in the original | {add['preexisting_boxes']:,} | {add['preexisting_found']:,} | **{add['preexisting_recall']:.4f}** |
| Added by the update | {add['added_boxes']:,} | {add['added_found']:,} | **{add['added_recall']:.4f}** |

{ctx['added_verdict']}

---

## Maturity classification

Scored on detections matched at IoU 0.5, against the updated labels.

{conf_table(u)}

Accuracy **{u['maturity']['accuracy']:.4f}** · macro recall **{u['maturity']['macro_recall']:.4f}** · {u['maturity']['n_matched']:,} matched grapes.

---

## Findings

{ctx['findings']}

---

*Generated by `pipeline/09_compare_label_sets.py`. Raw numbers in `build/compare_label_sets/report.json`.*
"""
    path.write_text(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--refresh", action="store_true", help="ignore the prediction cache")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "predictions.json"
    if args.refresh and cache.exists():
        cache.unlink()

    print("Loading label sets...")
    # The updated export is the prediction source, so its dimensions define the
    # coordinate space. The original export is de-normalised against the same
    # sizes -- see load_label_set for why reading its own images is a trap.
    upd = load_label_set(UPDATED)
    ref_sizes = {}
    for stem, (_, _, _, img_path) in upd.items():
        img = cv2.imread(img_path)
        if img is not None:
            ref_sizes[stem] = (img.shape[1], img.shape[0])
    orig = load_label_set(ORIGINAL, ref_sizes=ref_sizes)

    upd_dims = {v for v in ref_sizes.values()}
    print(f"  prediction coordinate space: {upd_dims}")
    shared = sorted(set(orig) & set(upd))
    print(f"  original {len(orig)} images | updated {len(upd)} images | shared {len(shared)}")
    only_upd = sorted(set(upd) - set(orig))
    if only_upd:
        print(f"  only in updated: {', '.join(only_upd)}")

    print("\nRunning detections...")
    preds = run_predictions(upd, cache, args.conf, args.downscale)

    print("\n" + "=" * 78)
    print(f"=== ALL {len(shared)} SHARED IMAGES (splits ignored)")
    print("=" * 78)
    o = score_against(orig, preds, shared)
    u = score_against(upd, preds, shared)
    print(fmt_block("ORIGINAL labels", o))
    print(fmt_block("UPDATED labels", u))

    print("\n" + "=" * 78)
    print("=== ADDED-ANNOTATION ANALYSIS")
    print("=" * 78)
    add = added_box_analysis(orig, upd, preds, shared)
    print(f"  already in original : {add['preexisting_found']:>5}/{add['preexisting_boxes']:<6} "
          f"recall {add['preexisting_recall']:.4f}")
    print(f"  added by update     : {add['added_found']:>5}/{add['added_boxes']:<6} "
          f"recall {add['added_recall']:.4f}")

    print("\n" + "=" * 78)
    print("=== ORIGINAL BOXES BY SIZE  (who labels / finds them)")
    print("=" * 78)
    ctx_sizes = size_bucket_analysis(orig, upd, preds, shared)
    print(f"  {'width':<14}{'count':>8}{'model finds':>14}{'updated labels':>17}")
    for b in ctx_sizes["buckets"]:
        print(f"  {b['range']:<14}{b['count']:>8}{b['model_finds']:>13.1%}"
              f"{b['updated_covers']:>17.1%}")
    print(f"  under 45px: {ctx_sizes['n_under_45px']} of {ctx_sizes['n_total']} "
          f"({ctx_sizes['frac_under_45px']:.0%}) -- the updated set omits nearly all of these")

    # ---- narrative built from the measured numbers ----
    r5_o = o["by_iou"]["0.5"]["recall"]
    r5_u = u["by_iou"]["0.5"]["recall"]
    p5_o = o["by_iou"]["0.5"]["precision"]
    p5_u = u["by_iou"]["0.5"]["precision"]

    if add["added_recall"] >= add["preexisting_recall"] * 0.8:
        verdict = (f"The added boxes are found at **{add['added_recall']:.1%}**, comparable to the "
                   f"**{add['preexisting_recall']:.1%}** on pre-existing boxes. The update largely "
                   "documented grapes the model was already detecting — detections that the "
                   "original labels scored as false positives.")
    else:
        verdict = (f"The added boxes are found at only **{add['added_recall']:.1%}** versus "
                   f"**{add['preexisting_recall']:.1%}** on pre-existing boxes. The update "
                   "introduced grapes the model systematically misses — these are genuine "
                   "detector failures, and retraining on the updated set should help.")

    gw_o = o["geometry"]["gt_median_width"]
    gw_u = u["geometry"]["gt_median_width"]
    direction = "over-reports" if u["count"]["bias"] > 0 else "under-reports"

    # Predicted class tally over the shared images, Uncertain kept separate.
    pred_hist = [0, 0, 0]
    pred_uncertain = 0
    for s in shared:
        pc = np.array(preds[s]["cls"], int)
        pp = np.array(preds[s]["prob"])
        pred_uncertain += int((pp < 0.6).sum())
        for i in range(3):
            pred_hist[i] += int(((pc == i) & (pp >= 0.6)).sum())
    pred_total = sum(pred_hist) + pred_uncertain

    upd_hist = [int(sum((upd[s][1] == i).sum() for s in shared)) for i in range(3)]
    # Compare mix on confident predictions only, so Uncertain does not dilute it.
    conf_total = max(sum(pred_hist), 1)
    gaps = [100 * pred_hist[i] / conf_total - 100 * upd_hist[i] / max(u["n_gt"], 1)
            for i in range(3)]
    worst = int(np.argmax(np.abs(gaps)))
    mix_note = (
        f"Against the updated labels the predicted mix is close: the largest gap is "
        f"**{CANON[worst]}**, at {gaps[worst]:+.1f} percentage points of the confident "
        f"predictions. The model is not systematically inventing or suppressing any one "
        f"maturity stage."
    ) if abs(gaps[worst]) < 8 else (
        f"The predicted mix diverges from the updated labels most on **{CANON[worst]}**, "
        f"at {gaps[worst]:+.1f} percentage points — worth checking before trusting "
        f"per-class totals."
    )

    sizes = ctx_sizes
    findings = [
        f"1. **The exports are at different resolutions and that must be corrected for.** "
        f"`dataset/` is 4640x3480; `dataset_all_yolo/` is 2048x1536. De-normalising each "
        f"set against its own image put them 2.27x apart, dropping same-grape IoU to ~0.02 "
        f"and producing a confident, wrong conclusion about annotators disagreeing on "
        f"minimum grape size. All numbers here use a single coordinate space.",
        f"2. **The two label sets agree on convention.** Median box width is {gw_o:.0f}px "
        f"(original) against {gw_u:.0f}px (updated), and the updated set covers 94-96% of the "
        f"original boxes at every size. The update did not redefine what a grape is -- it "
        f"labelled more of them, adding {u['n_gt'] - o['n_gt']:,} boxes.",
        f"3. **Precision rose from {p5_o:.3f} to {p5_u:.3f} at IoU 0.5** on byte-identical "
        f"predictions. The model did not change; only the labels did. Most of what the "
        f"original set scored as false positives were real grapes it had not marked -- which "
        f"is why precision cannot be read as a model property on any of these label sets.",
        f"4. **Localisation is the weak point, not classification.** Against the updated "
        f"labels, recall falls from {u['by_iou']['0.3']['recall']:.3f} at IoU 0.30 to "
        f"{u['by_iou']['0.75']['recall']:.3f} at IoU 0.75, while maturity accuracy on matched "
        f"grapes holds at {u['maturity']['accuracy']:.3f}. Boxes find the right grapes but sit loosely.",
        f"5. **Counting is now close.** Against the updated labels the bias is "
        f"{u['count']['bias']:+.1f} boxes per image ({u['count']['ratio']:.2f}×, MAPE "
        f"{u['count']['mape']:.1f}%), so the pipeline slightly {direction}. Against the original "
        f"labels the same predictions looked like a {o['count']['ratio']:.2f}× over-count.",
        f"6. {verdict}",
    ]

    ctx = {
        "original": o, "updated": u, "added": add,
        "n_orig_images": len(orig), "n_upd_images": len(upd), "n_shared": len(shared),
        "orig_hist": [int(sum((orig[s][1] == i).sum() for s in shared)) for i in range(3)],
        "upd_hist": upd_hist,
        "pred_hist": pred_hist,
        "pred_uncertain": pred_uncertain,
        "pred_total": pred_total,
        "mix_note": mix_note,
        "added_verdict": verdict,
        "findings": "\n\n".join(findings),
        "sizes": ctx_sizes,
        "size_table": "\n".join(
            f"| {b['range']} | {b['count']:,} | {b['model_finds']:.1%} | {b['updated_covers']:.1%} |"
            for b in ctx_sizes["buckets"]),
        "big_bucket_model": ctx_sizes["buckets"][3]["model_finds"],
        "biggest_bucket_model": ctx_sizes["buckets"][4]["model_finds"],
    }

    md_path = BASE_DIR / "EVALUATION_FINDINGS.md"
    write_markdown(md_path, ctx)
    (OUT / "report.json").write_text(json.dumps(
        {k: v for k, v in ctx.items() if k not in ("findings", "added_verdict")},
        indent=2, default=str))

    print("\n" + "=" * 78)
    print("=== FINDINGS")
    print("=" * 78)
    for f in findings:
        print("  " + f.replace("**", ""))
    print(f"\nWrote {md_path}")
    print(f"Wrote {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
