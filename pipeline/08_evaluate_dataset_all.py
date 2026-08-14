"""
Step 8: Evaluate the trained pipeline against a ground-truth dataset.

Pass any dataset root with --dataset; the layout is detected automatically.

  YOLO EXPORT  (dataset/, dataset_all_yolo/)
      data.yaml plus train|valid|test/{images,labels}.
      Splits come from the folders themselves.

      Class ids are remapped through the export's OWN data.yaml. This is
      required, not defensive -- the two exports disagree about ordering:
          dataset_all_yolo : ['Darkening','Harvestable','Whitening','RedSlime']
          dataset          : ['RedSlime','Darkening','Harvestable','Whitening']
      Reading raw ids would mislabel every box while still producing plausible
      looking output.

      A label row with 4 coordinates is a box; more than 4 is a polygon, which
      is how RedSlime is stored.

  FLAT JSON  (dataset_all/)
      <stem>.jpg beside <stem>.json, with two formats mixed in one directory:
        * LabelMe dict  (27 files) -- shapes[] with points [[x1,y1],[x2,y2]]
        * CreateML list (20 files) -- annotations[] with coordinates {x,y,w,h},
          where x,y is the box CENTRE (verified at IoU 0.965 against the
          Roboflow boxes; the top-left reading scores 0.152).
      Splits are recovered by looking each stem up in build/detect/.

How to read the output, whichever dataset is used:

  * RECALL is the headline. "Of the grapes a person marked, how many did we find?"
  * PRECISION is a floor, not an estimate. No label set here is exhaustive, so a
    detection with no matching ground-truth box is usually a real grape nobody
    marked. Scoring identical predictions against dataset_all_yolo/ versus
    dataset/ moves precision from 0.08 to 0.71 without the model changing.

Output goes to build/eval_<dataset name>/ so runs against different datasets do
not overwrite each other.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import TwoStageDetector  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DETECT = BASE_DIR / "build" / "detect"

# Matches the classifier's ImageFolder ordering (alphabetical).
CLASS_NAMES = ["Darkening", "Harvestable", "Whitening"]
CLASS_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

SPLITS = ["train", "valid", "test"]


# ---------------------------------------------------------------- label input

def yaml_names(path):
    """Minimal reader for the `names: [...]` line; avoids a pyyaml dependency."""
    for line in path.read_text().splitlines():
        if line.strip().startswith("names:"):
            raw = line.split("names:", 1)[1].strip()
            return [n.strip().strip("'\"") for n in raw.strip("[]").split(",")]
    raise ValueError(f"no names: line in {path}")


def parse_json_annotation(path):
    """Return (boxes xyxy, class ids, slime polygons) from either JSON format."""
    data = json.loads(path.read_text())
    boxes, classes, polygons = [], [], []

    if isinstance(data, dict) and "shapes" in data:          # LabelMe
        for s in data["shapes"]:
            pts = s["points"]
            if s["shape_type"] == "rectangle":
                (x1, y1), (x2, y2) = pts[0], pts[1]
                box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            elif s["shape_type"] == "polygon":
                if s["label"] == "RedSlime":
                    polygons.append(np.array(pts, np.float32))
                continue
            else:
                continue
            if s["label"] in CLASS_IDX:
                boxes.append(box)
                classes.append(CLASS_IDX[s["label"]])

    elif isinstance(data, list):                              # CreateML
        for entry in data:
            for a in entry.get("annotations", []):
                if a["label"] not in CLASS_IDX:
                    continue
                c = a["coordinates"]
                # x,y is the centre -- verified empirically, see module docstring.
                boxes.append([c["x"] - c["width"] / 2, c["y"] - c["height"] / 2,
                              c["x"] + c["width"] / 2, c["y"] + c["height"] / 2])
                classes.append(CLASS_IDX[a["label"]])

    return np.array(boxes).reshape(-1, 4), np.array(classes, int), polygons


def parse_yolo_annotation(path, names, W, H):
    """Return (boxes xyxy, class ids, slime polygons) from a YOLO .txt label file.

    Coordinates are normalised, so the image dimensions are required. Row length
    is the only thing separating a box from a polygon.
    """
    boxes, classes, polygons = [], [], []
    if not path.exists():
        return np.zeros((0, 4)), np.zeros((0,), int), polygons

    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        name = names[cid] if 0 <= cid < len(names) else None
        coords = [float(v) for v in parts[1:]]

        if len(coords) == 4:
            if name not in CLASS_IDX:      # RedSlime drawn as a rectangle
                continue
            cx, cy, bw, bh = coords[0] * W, coords[1] * H, coords[2] * W, coords[3] * H
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            classes.append(CLASS_IDX[name])
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            if name == "RedSlime":
                polygons.append((np.array(coords).reshape(-1, 2) * [W, H]).astype(np.float32))

    return np.array(boxes).reshape(-1, 4), np.array(classes, int), polygons


def split_of_stem(stem):
    """Recover a split for flat-JSON datasets by looking the stem up in build/detect."""
    for split in SPLITS:
        if list((DETECT / split / "images").glob(stem + "_*")):
            return split
    return "unknown"


def collect_records(root):
    """Return (records, layout). Each record locates one image and its labels.

    Labels are not parsed here: YOLO coordinates are normalised and need the
    image dimensions, which are only known once the image is read.
    """
    is_yolo = (root / "data.yaml").exists() and (root / "train").is_dir()

    records = []
    if is_yolo:
        names = yaml_names(root / "data.yaml")
        for split in SPLITS:
            img_dir = root / split / "images"
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*")):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                records.append({
                    # .stem first: dataset_all_yolo names files "002.jpg" with no
                    # underscore, so splitting the full name would keep ".jpg".
                    "stem": img_path.stem.split("_")[0],
                    "split": split,
                    "img_path": img_path,
                    "label_path": root / split / "labels" / (img_path.stem + ".txt"),
                    "layout": "yolo",
                    "names": names,
                })
        return records, f"YOLO export, classes {names}"

    for img_path in sorted(root.glob("*.jpg")):
        json_path = root / f"{img_path.stem}.json"
        if not json_path.exists():
            continue
        records.append({
            "stem": img_path.stem,
            "split": split_of_stem(img_path.stem),
            "img_path": img_path,
            "label_path": json_path,
            "layout": "json",
            "names": None,
        })
    return records, "flat JSON (LabelMe / CreateML)"


# ---------------------------------------------------------------- matching

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


def greedy_match(pred, gt, scores, thr):
    """Highest-confidence-first assignment of predictions to ground truth."""
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
    """Match when a prediction's centre lands inside a ground-truth box.

    Reported alongside IoU because the label sets use different box scales, and
    IoU conflates "found the wrong thing" with "drew a differently sized box
    around the right thing".
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


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset_all",
                    help="dataset root, relative to the project (default: dataset_all). "
                         "Accepts a YOLO export or a flat JSON directory.")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--save-vis", action="store_true")
    ap.add_argument("--box-thickness", type=int, default=0,
                    help="bounding box line width in px; 0 derives it from image width")
    ap.add_argument("--gt-overlay", action="store_true",
                    help="also draw ground-truth boxes, for side-by-side comparison")
    ap.add_argument("--no-slime", action="store_true",
                    help="omit the RedSlime tint and its panel line from the rendered images; "
                         "the RedSlime coverage metrics are still reported")
    ap.add_argument("--slime-metrics-off", action="store_true",
                    help="also skip the RedSlime coverage evaluation entirely")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N images")
    args = ap.parse_args()

    src = BASE_DIR / args.dataset
    if not src.exists():
        raise SystemExit(f"dataset not found: {src}")

    out = BASE_DIR / "build" / f"eval_{src.name}"
    out.mkdir(parents=True, exist_ok=True)

    records, layout = collect_records(src)
    if not records:
        raise SystemExit(f"no image/label pairs found in {src}")
    if args.limit:
        records = records[:args.limit]

    print(f"Dataset : {src}")
    print(f"Layout  : {layout}")
    print(f"Images  : {len(records)}")
    print(f"Output  : {out}\n")

    det = TwoStageDetector(conf=args.conf)

    agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n_img": 0,
                               "conf": np.zeros((3, 4), int), "right": 0, "total": 0,
                               "ctp": 0, "cfn": 0})
    rows, slime_rows = [], []

    for rec in records:
        img = cv2.imread(str(rec["img_path"]))
        if img is None:
            print(f"  ! unreadable {rec['img_path'].name}")
            continue
        H, W = img.shape[:2]
        stem, split = rec["stem"], rec["split"]

        if rec["layout"] == "yolo":
            gt_boxes, gt_cls, polys = parse_yolo_annotation(
                rec["label_path"], rec["names"], W, H)
        else:
            gt_boxes, gt_cls, polys = parse_json_annotation(rec["label_path"])

        boxes, scores = det.detect(img, downscale=args.downscale)
        pred_cls, pred_prob = det.classify(img, boxes)
        pb, sc = boxes.numpy(), scores.numpy()

        assign = greedy_match(pb, gt_boxes, sc, args.iou)
        matched = int((assign >= 0).sum())
        c_matched = int((match_centre(pb, gt_boxes, sc) >= 0).sum())

        a = agg[split]
        a["n_img"] += 1
        a["tp"] += matched
        a["fp"] += len(pb) - matched
        a["fn"] += len(gt_boxes) - matched
        a["ctp"] += c_matched
        a["cfn"] += len(gt_boxes) - c_matched

        for pi, gi in enumerate(assign):
            if gi < 0:
                continue
            truth = gt_cls[gi]
            a["total"] += 1
            if pred_prob[pi] < det.uncertain_below:
                a["conf"][truth, 3] += 1
            else:
                a["conf"][truth, pred_cls[pi]] += 1
                if pred_cls[pi] == truth:
                    a["right"] += 1

        gt_counts = np.bincount(gt_cls, minlength=3)
        pr_counts = np.bincount(pred_cls[pred_prob >= det.uncertain_below], minlength=3)
        rows.append({
            "stem": stem, "split": split,
            "gt": int(len(gt_boxes)), "pred": int(len(pb)),
            "matched": matched, "centre_matched": c_matched,
            "recall": matched / len(gt_boxes) if len(gt_boxes) else float("nan"),
            "gt_counts": gt_counts.tolist(), "pred_counts": pr_counts.tolist(),
        })

        if polys and not args.slime_metrics_off:
            gt_mask = np.zeros((H, W), np.uint8)
            for p in polys:
                cv2.fillPoly(gt_mask, [p.astype(np.int32)], 1)
            cov, mask = det.slime_coverage(img)
            pred_full = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) > 0
            inter = np.logical_and(pred_full, gt_mask > 0).sum()
            union = np.logical_or(pred_full, gt_mask > 0).sum()
            slime_rows.append({"stem": stem, "gt_cov": float(gt_mask.mean()),
                               "pred_cov": float(cov),
                               "iou": float(inter / union) if union else 1.0})

        if args.save_vis:
            dets = [{"box": b.tolist(),
                     "class": CLASS_NAMES[c] if p >= det.uncertain_below else "Uncertain",
                     "classifier_conf": float(p), "detector_conf": float(s)}
                    for b, c, p, s in zip(pb, pred_cls, pred_prob, sc)]
            # Passing None for both suppresses the magenta tint and drops the
            # "RedSlime: x%" line from the summary panel; grape boxes are
            # unaffected, since RedSlime was never one of the three classes.
            cov, mask = (None, None) if args.no_slime else det.slime_coverage(img)
            vis = det.draw(img, dets, mask, cov, box_thickness=args.box_thickness or None)

            if args.gt_overlay:
                # Ground truth in cyan, heavier and on top, so the two box
                # conventions can be compared directly.
                t = args.box_thickness or max(2, int(W / 600))
                for (x1, y1, x2, y2) in gt_boxes.astype(int):
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), t + 2)

            cv2.imwrite(str(out / f"{stem}_{split}_pred.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

        r = rows[-1]
        print(f"  {stem} [{split:<5}] gt {r['gt']:>4} | found {r['pred']:>4} | "
              f"matched {matched:>4} | recall {r['recall']:.3f}")

    # ---------------- report ----------------
    def block(title, splits):
        t = {"tp": 0, "fp": 0, "fn": 0, "n_img": 0, "right": 0, "total": 0,
             "ctp": 0, "cfn": 0, "conf": np.zeros((3, 4), int)}
        for s in splits:
            if s not in agg:
                continue
            for k in ["tp", "fp", "fn", "n_img", "right", "total", "ctp", "cfn"]:
                t[k] += agg[s][k]
            t["conf"] += agg[s]["conf"]
        if t["n_img"] == 0:
            return None
        rec = t["tp"] / (t["tp"] + t["fn"]) if (t["tp"] + t["fn"]) else 0.0
        prec = t["tp"] / (t["tp"] + t["fp"]) if (t["tp"] + t["fp"]) else 0.0
        crec = t["ctp"] / (t["ctp"] + t["cfn"]) if (t["ctp"] + t["cfn"]) else 0.0
        print(f"\n{title}  ({t['n_img']} images)")
        print(f"  ground-truth grapes   : {t['tp'] + t['fn']}")
        print(f"  detections            : {t['tp'] + t['fp']}")
        print(f"  RECALL @ IoU {args.iou:<4}    : {rec:.4f}   <- headline")
        print(f"  recall, centre match  : {crec:.4f}   <- ignores box-size convention")
        print(f"  precision             : {prec:.4f}   <- floor only, see note")
        if t["total"]:
            print(f"  maturity accuracy     : {t['right'] / t['total']:.4f} "
                  f"(on {t['total']} matched grapes)")
        return t

    print("\n" + "=" * 74)
    print(f"=== DETECTION vs GROUND TRUTH  ({src.name})")
    print("=" * 74)
    for name, splits in [("TEST SPLIT -- never seen in training", ["test"]),
                         ("VALID SPLIT", ["valid"]),
                         ("TRAIN SPLIT -- seen in training, optimistic", ["train"]),
                         ("UNKNOWN SPLIT", ["unknown"])]:
        block(name, splits)
    block(f"ALL {len(rows)} IMAGES", SPLITS + ["unknown"])

    # Maturity confusion, preferring the held-out split when it exists.
    conf_split = "test" if ("test" in agg and agg["test"]["total"]) else None
    conf = agg[conf_split]["conf"] if conf_split else sum(
        (agg[s]["conf"] for s in agg), np.zeros((3, 4), int))
    if conf.sum():
        label = f"{conf_split} split" if conf_split else "all images"
        print("\n" + "=" * 74)
        print(f"=== MATURITY, {label}, on matched grapes (rows = ground truth)")
        print("=" * 74)
        print(f"{'':<14}" + "".join(f"{n[:11]:>12}" for n in CLASS_NAMES)
              + f"{'Uncertain':>12}{'recall':>9}")
        recalls = []
        for i, n in enumerate(CLASS_NAMES):
            tot = conf[i].sum()
            r = conf[i, i] / tot if tot else 0.0
            recalls.append(r)
            print(f"{n:<14}" + "".join(f"{v:>12}" for v in conf[i]) + f"{r:>9.3f}")
        print(f"  macro recall: {np.mean(recalls):.4f}")

    print("\n" + "=" * 74)
    print("=== PER-IMAGE COUNTS")
    print("=" * 74)
    print(f"  {'img':<6}{'split':<7}{'gt':>7}{'found':>7}{'matched':>9}"
          f"{'centre':>8}{'recall':>9}{'ratio':>8}")
    for r in sorted(rows, key=lambda x: (x["split"], x["stem"])):
        ratio = r["pred"] / r["gt"] if r["gt"] else float("nan")
        print(f"  {r['stem']:<6}{r['split']:<7}{r['gt']:>7}{r['pred']:>7}"
              f"{r['matched']:>9}{r['centre_matched']:>8}{r['recall']:>9.3f}{ratio:>8.1f}x")

    if slime_rows:
        print("\n" + "=" * 74)
        print("=== REDSLIME COVERAGE")
        print("=" * 74)
        print(f"  {'img':<6}{'truth %':>10}{'predicted %':>14}{'IoU':>8}")
        for s in slime_rows:
            print(f"  {s['stem']:<6}{100 * s['gt_cov']:>10.1f}"
                  f"{100 * s['pred_cov']:>14.1f}{s['iou']:>8.3f}")

    print("\n" + "=" * 74)
    print("=== HOW TO READ THIS")
    print("=" * 74)
    print("  No label set here is exhaustive, so a detection with no matching")
    print("  ground-truth box is usually a real grape nobody marked. Precision is")
    print("  a floor that mostly measures annotation coverage -- the same")
    print("  predictions score 0.08 against dataset_all_yolo and 0.71 against")
    print("  dataset. Recall is the number that means something, and the centre-")
    print("  match variant removes the box-size convention from the comparison.")

    (out / "report.json").write_text(json.dumps({
        "dataset": str(src), "layout": layout,
        "conf": args.conf, "iou": args.iou,
        "per_image": rows,
        "slime": slime_rows,
        "by_split": {k: {"tp": v["tp"], "fp": v["fp"], "fn": v["fn"], "n_img": v["n_img"],
                         "centre_tp": v["ctp"], "centre_fn": v["cfn"],
                         "maturity_right": v["right"], "maturity_total": v["total"],
                         "confusion": v["conf"].tolist()} for k, v in agg.items()},
    }, indent=2))
    print(f"\nWrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
