"""
Step 10: Render the ground-truth labels of a YOLO dataset onto its images.

This draws what the ANNOTATOR marked, not what the model predicts, using the
same visual language as the prediction renders in 08_evaluate_dataset_all.py --
same class colours, same box weight, same summary panel -- so the two can be
compared side by side without the styling itself being a variable.

No model is loaded. This is cv2 and numpy only.

Each dataset is rendered separately into its own directory, because the whole
point is to see the two annotation conventions apart from each other:

    build/rendered_labels/dataset/            updated labels
    build/rendered_labels/dataset_all_yolo/   original labels

Class ids are read through each export's own data.yaml, which matters here:
the two files disagree about ordering.

    dataset_all_yolo : ['Darkening', 'Harvestable', 'Whitening', 'RedSlime']
    dataset          : ['RedSlime', 'Darkening', 'Harvestable', 'Whitening']

Reading raw ids without that remap would colour every box wrong while still
producing a plausible-looking picture.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_ROOT = BASE_DIR / "build" / "rendered_labels"

# BGR, matching predict.py so label renders and prediction renders are comparable.
COLORS = {
    "Darkening": (34, 126, 230),     # orange
    "Harvestable": (113, 204, 46),   # green
    "Whitening": (255, 255, 255),    # white
    "RedSlime": (180, 90, 220),      # magenta
}
FALLBACK = (0, 255, 0)

BOX_ALPHA = 0.75


def yaml_names(path):
    """Minimal reader for the `names: [...]` line; avoids a pyyaml dependency."""
    for line in path.read_text().splitlines():
        if line.strip().startswith("names:"):
            raw = line.split("names:", 1)[1].strip()
            return [n.strip().strip("'\"") for n in raw.strip("[]").split(",")]
    raise ValueError(f"no names: line in {path}")


def parse_label_file(path, names, W, H):
    """Return (boxes, polygons) in absolute pixels, each tagged with a class name.

    A YOLO row is `cls cx cy w h` for a box, or `cls x1 y1 x2 y2 ...` for a
    polygon. Length is the only thing that distinguishes them.
    """
    boxes, polygons = [], []
    if not path.exists():
        return boxes, polygons

    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        name = names[cid] if 0 <= cid < len(names) else str(cid)
        coords = [float(v) for v in parts[1:]]

        if len(coords) == 4:
            cx, cy, bw, bh = coords[0] * W, coords[1] * H, coords[2] * W, coords[3] * H
            boxes.append((name, int(cx - bw / 2), int(cy - bh / 2),
                          int(cx + bw / 2), int(cy + bh / 2)))
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            pts = (np.array(coords).reshape(-1, 2) * [W, H]).astype(np.int32)
            polygons.append((name, pts))
    return boxes, polygons


def render(img, boxes, polygons, title, thickness=None):
    """Draw labels over the image and stamp a per-class tally panel."""
    H, W = img.shape[:2]
    out = img.copy()
    t = thickness or max(2, int(W / 600))

    # Polygons first, as a translucent fill -- they are regions, not objects,
    # and would otherwise hide the boxes drawn on top of them.
    if polygons:
        overlay = out.copy()
        for name, pts in polygons:
            cv2.fillPoly(overlay, [pts], COLORS.get(name, FALLBACK))
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
        for name, pts in polygons:
            cv2.polylines(out, [pts], True, COLORS.get(name, FALLBACK), t + 2)

    overlay = out.copy()
    for name, x1, y1, x2, y2 in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS.get(name, FALLBACK), t)
    cv2.addWeighted(overlay, BOX_ALPHA, out, 1 - BOX_ALPHA, 0, out)

    tally = {}
    for name, *_ in boxes:
        tally[name] = tally.get(name, 0) + 1
    for name, _ in polygons:
        tally[f"{name} (poly)"] = tally.get(f"{name} (poly)", 0) + 1

    lines = [title, f"Total: {len(boxes)}"] + [f"{k}: {v}" for k, v in sorted(tally.items())]

    scale = max(0.6, W / 2200)
    pad, lh = int(14 * scale), int(34 * scale)
    box_w = int(420 * scale)
    cv2.rectangle(out, (pad, pad), (pad + box_w, pad + lh * len(lines) + pad), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        key = line.split(":")[0].replace(" (poly)", "")
        colour = COLORS.get(key, (255, 255, 255))
        cv2.putText(out, line, (pad * 2, pad + lh * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, colour, max(1, int(2 * scale)))
    return out


def render_dataset(root, out_dir, thickness, downscale, quality, drop_slime=False):
    names = yaml_names(root / "data.yaml")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {root.name}")
    print(f"    classes: {names}")

    totals, n_images, n_boxes, n_polys = {}, 0, 0, 0

    for split in ["train", "valid", "test"]:
        img_dir = root / split / "images"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    ! unreadable {img_path.name}")
                continue
            H, W = img.shape[:2]
            stem = img_path.name.split("_")[0].replace(".jpg", "").replace(".png", "")

            boxes, polygons = parse_label_file(
                root / split / "labels" / (img_path.stem + ".txt"), names, W, H)
            if drop_slime:
                # RedSlime can also appear as a rectangle row, so filter both.
                boxes = [b for b in boxes if b[0] != "RedSlime"]
                polygons = []

            vis = render(img, boxes, polygons, f"{root.name} / {stem} [{split}]", thickness)
            if downscale > 1:
                vis = cv2.resize(vis, (W // downscale, H // downscale),
                                 interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{stem}_{split}.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])

            for name, *_ in boxes:
                totals[name] = totals.get(name, 0) + 1
            n_images += 1
            n_boxes += len(boxes)
            n_polys += len(polygons)
            print(f"    {stem} [{split:<5}] {len(boxes):>4} boxes"
                  + (f"  {len(polygons)} polygons" if polygons else ""))

    print(f"    -> {n_images} images | {n_boxes} boxes | {n_polys} polygons")
    print(f"    -> {'  '.join(f'{k}={v}' for k, v in sorted(totals.items()))}")
    print(f"    -> written to {out_dir}")
    return {"images": n_images, "boxes": n_boxes, "polygons": n_polys, "by_class": totals}


def main():
    ap = argparse.ArgumentParser(description="Render YOLO ground-truth labels onto images")
    ap.add_argument("--datasets", nargs="+", default=["dataset", "dataset_all_yolo"],
                    help="dataset directories, relative to the project root")
    ap.add_argument("--box-thickness", type=int, default=0,
                    help="line width in px; 0 derives it from image width")
    ap.add_argument("--downscale", type=int, default=1,
                    help="shrink the written image by this factor")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--no-slime", action="store_true",
                    help="omit RedSlime polygons from the rendered images")
    args = ap.parse_args()

    summary = {}
    for name in args.datasets:
        root = BASE_DIR / name
        if not (root / "data.yaml").exists():
            print(f"! skipping {name}: no data.yaml")
            continue
        summary[name] = render_dataset(root, OUT_ROOT / name,
                                       args.box_thickness or None,
                                       args.downscale, args.quality,
                                       drop_slime=args.no_slime)

    print("\n" + "=" * 70)
    print("=== SUMMARY")
    print("=" * 70)
    classes = sorted({c for s in summary.values() for c in s["by_class"]})
    print(f"  {'dataset':<22}{'images':>8}{'boxes':>8}" + "".join(f"{c[:11]:>13}" for c in classes))
    for name, s in summary.items():
        print(f"  {name:<22}{s['images']:>8}{s['boxes']:>8}"
              + "".join(f"{s['by_class'].get(c, 0):>13}" for c in classes))


if __name__ == "__main__":
    main()
