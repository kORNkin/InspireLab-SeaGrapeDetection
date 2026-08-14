"""
Step 1: Clean and split the Roboflow export.

The exported label files mix two annotation types in the same .txt:
  - 5-field lines  -> axis-aligned grape bounding boxes (classes 1,2,3)
  - N-field lines  -> RedSlime polygons (class 0), YOLO-seg format

Ultralytics detect mode cannot parse the polygon lines, so they are separated
into their own tree. Grape classes are also remapped to a contiguous 0..2.

Source class ids -> output:
    0 'RedSlime'    -> build/slime/<split>/{images,labels}  (polygons kept as-is)
    1 'Darkening'   -> 0
    2 'Harvestable' -> 1
    3 'Whitening'   -> 2

Images 029 and 030 are flagged: their grapes inside slime regions were never
annotated, so they would teach the detector to suppress real grapes. They are
copied but recorded in build/detect/excluded.txt for step 2 to skip.
"""

import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC = BASE_DIR / "dataset"
OUT = BASE_DIR / "build"

SPLITS = ["train", "valid", "test"]

# Source id -> new contiguous id for the grape detector/classifier
GRAPE_REMAP = {1: 0, 2: 1, 3: 2}
GRAPE_NAMES = ["Darkening", "Harvestable", "Whitening"]

SLIME_SRC_ID = 0
SLIME_NAMES = ["RedSlime"]

# Under-annotated relative to the rest of the set, verified by eye:
#   029, 030 - grapes inside the RedSlime patches were left unlabelled
#   002      - a large in-focus grape field on the right was skipped entirely
EXCLUDE_FROM_DETECTOR = {"029", "030", "002"}


def stem_of(path):
    """Roboflow mangles names to `029_jpg.rf.<hash>.jpg`; recover the `029`."""
    return path.name.split("_")[0]


def parse_label_file(path):
    """Return (boxes, polygons). boxes: (cls, cx, cy, w, h). polygons: (cls, [x,y,...])."""
    boxes, polygons = [], []
    for raw in path.read_text().splitlines():
        parts = raw.split()
        if not parts:
            continue
        cls = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        if len(coords) == 4:
            boxes.append((cls, *coords))
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            polygons.append((cls, coords))
        else:
            raise ValueError(f"{path.name}: unparseable line with {len(coords)} coords")
    return boxes, polygons


def write_yaml(path, names):
    lines = [
        "train: ../train/images",
        "val: ../valid/images",
        "test: ../test/images",
        "",
        f"nc: {len(names)}",
        "names: " + json.dumps(names),
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    if OUT.exists():
        shutil.rmtree(OUT)

    detect_root = OUT / "detect"
    slime_root = OUT / "slime"

    for root in (detect_root, slime_root):
        for split in SPLITS:
            (root / split / "images").mkdir(parents=True, exist_ok=True)
            (root / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {s: {"images": 0, "boxes": [0, 0, 0], "dropped": 0} for s in SPLITS}
    slime_stats = {s: {"images": 0, "polygons": 0} for s in SPLITS}
    per_image_counts = []

    for split in SPLITS:
        for img_path in sorted((SRC / split / "images").glob("*")):
            label_path = SRC / split / "labels" / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"  ! orphan image, skipped: {img_path.name}")
                continue

            boxes, polygons = parse_label_file(label_path)
            stem = stem_of(img_path)

            # --- grape detection/classification tree ---
            kept = []
            for cls, cx, cy, w, h in boxes:
                if cls not in GRAPE_REMAP:
                    stats[split]["dropped"] += 1
                    continue
                new_cls = GRAPE_REMAP[cls]
                stats[split]["boxes"][new_cls] += 1
                kept.append(f"{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            shutil.copy2(img_path, detect_root / split / "images" / img_path.name)
            (detect_root / split / "labels" / f"{img_path.stem}.txt").write_text(
                "\n".join(kept) + ("\n" if kept else "")
            )
            stats[split]["images"] += 1
            per_image_counts.append((split, stem, len(kept)))

            # --- slime polygon tree (only images that actually have polygons) ---
            slime_lines = [
                "0 " + " ".join(f"{v:.6f}" for v in coords)
                for cls, coords in polygons
                if cls == SLIME_SRC_ID
            ]
            if slime_lines:
                shutil.copy2(img_path, slime_root / split / "images" / img_path.name)
                (slime_root / split / "labels" / f"{img_path.stem}.txt").write_text(
                    "\n".join(slime_lines) + "\n"
                )
                slime_stats[split]["images"] += 1
                slime_stats[split]["polygons"] += len(slime_lines)

    write_yaml(detect_root / "data.yaml", GRAPE_NAMES)
    write_yaml(slime_root / "data.yaml", SLIME_NAMES)

    # Record which source images step 2 must not tile into detector training data.
    excluded = sorted(EXCLUDE_FROM_DETECTOR)
    (detect_root / "excluded.txt").write_text("\n".join(excluded) + "\n")

    # --- report ---
    print("\n=== Grape detection tree -> build/detect ===")
    total = [0, 0, 0]
    for split in SPLITS:
        s = stats[split]
        total = [a + b for a, b in zip(total, s["boxes"])]
        named = "  ".join(f"{n}={c}" for n, c in zip(GRAPE_NAMES, s["boxes"]))
        print(
            f"  {split:<6} {s['images']:>3} images | {sum(s['boxes']):>5} boxes | "
            f"{named} | dropped non-grape lines: {s['dropped']}"
        )
    print(f"  {'TOTAL':<6} {'':>3}        | {sum(total):>5} boxes | "
          + "  ".join(f"{n}={c}" for n, c in zip(GRAPE_NAMES, total)))

    print("\n=== RedSlime polygon tree -> build/slime ===")
    for split in SPLITS:
        s = slime_stats[split]
        print(f"  {split:<6} {s['images']:>3} images | {s['polygons']:>3} polygons")

    print(f"\n=== Excluded from stage-1 tiling (under-annotated): {', '.join(excluded)} ===")

    # Surface any other suspiciously sparse images so they can be checked by hand.
    counts = sorted(per_image_counts, key=lambda r: r[2])
    print("\n=== 8 sparsest images (verify these are not under-annotated) ===")
    for split, stem, n in counts[:8]:
        flag = "  <-- excluded" if stem in EXCLUDE_FROM_DETECTOR else ""
        print(f"  {split:<6} {stem}  {n:>4} boxes{flag}")


if __name__ == "__main__":
    main()
