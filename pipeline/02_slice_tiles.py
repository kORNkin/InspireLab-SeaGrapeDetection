"""
Step 2: Slice the 4640x3480 images into 640x640 tiles for the stage-1 detector.

Why tile: a grape is ~80x79 px. Feeding the full image to YOLO at imgsz=640
shrinks it to ~11 px and destroys it. At 640 tiles the grape stays 80 px.

Why stride 512: the 128 px overlap exceeds the 80 px grape width, so every
grape appears whole inside at least one tile.

All three grape classes collapse to a single class here. Stage 1 only has to
answer "where is a grape"; the maturity call belongs to stage 2, where the
13:1 class imbalance can be handled with a weighted sampler.

Empty tiles are dropped rather than kept as background negatives. The
annotators labelled only a subset of the resolvable spheres, so a tile with no
boxes very likely still contains real unlabelled grapes -- training on it as
pure background would teach the detector to suppress them.
"""

import shutil
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
SRC = BASE_DIR / "build" / "detect"
OUT = BASE_DIR / "build" / "tiles"

SPLITS = ["train", "valid", "test"]

TILE = 640
STRIDE = 512

# A box clipped by a tile edge is kept only if most of it survives; the
# overlap guarantees the same grape is intact in a neighbouring tile.
MIN_VISIBLE_FRACTION = 0.6

# Degenerate slivers help nobody regardless of visible fraction.
MIN_BOX_PX = 8


def tile_offsets(total, tile, stride):
    """Offsets covering [0, total) with the final tile flush against the edge."""
    if total <= tile:
        return [0]
    offs = list(range(0, total - tile + 1, stride))
    if offs[-1] != total - tile:
        offs.append(total - tile)
    return offs


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in SPLITS:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    excluded = set((SRC / "excluded.txt").read_text().split())

    totals = {}
    for split in SPLITS:
        n_tiles = n_empty = n_boxes = n_src = 0

        for img_path in sorted((SRC / split / "images").glob("*")):
            stem = img_path.name.split("_")[0]
            if stem in excluded:
                print(f"  skip (under-annotated): {split}/{stem}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ! unreadable: {img_path.name}")
                continue
            H, W = img.shape[:2]
            n_src += 1

            # Absolute-pixel boxes for this image.
            label_path = SRC / split / "labels" / (img_path.stem + ".txt")
            abs_boxes = []
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                _, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H
                abs_boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))

            for oy in tile_offsets(H, TILE, STRIDE):
                for ox in tile_offsets(W, TILE, STRIDE):
                    rows = []
                    for x1, y1, x2, y2 in abs_boxes:
                        # Reject before clipping so we can measure what was lost.
                        if x2 <= ox or x1 >= ox + TILE or y2 <= oy or y1 >= oy + TILE:
                            continue
                        full_area = (x2 - x1) * (y2 - y1)
                        cx1, cy1 = max(x1, ox), max(y1, oy)
                        cx2, cy2 = min(x2, ox + TILE), min(y2, oy + TILE)
                        cw, ch = cx2 - cx1, cy2 - cy1
                        if cw < MIN_BOX_PX or ch < MIN_BOX_PX:
                            continue
                        if full_area <= 0 or (cw * ch) / full_area < MIN_VISIBLE_FRACTION:
                            continue
                        rows.append(
                            f"0 {((cx1 + cx2) / 2 - ox) / TILE:.6f} "
                            f"{((cy1 + cy2) / 2 - oy) / TILE:.6f} "
                            f"{cw / TILE:.6f} {ch / TILE:.6f}"
                        )

                    if not rows:
                        n_empty += 1
                        continue

                    name = f"{stem}_{ox:05d}_{oy:05d}"
                    cv2.imwrite(
                        str(OUT / split / "images" / f"{name}.jpg"),
                        img[oy:oy + TILE, ox:ox + TILE],
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )
                    (OUT / split / "labels" / f"{name}.txt").write_text("\n".join(rows) + "\n")
                    n_tiles += 1
                    n_boxes += len(rows)

        totals[split] = (n_src, n_tiles, n_empty, n_boxes)
        print(
            f"  {split:<6} {n_src:>2} images -> {n_tiles:>4} tiles kept "
            f"({n_empty} empty dropped) | {n_boxes} boxes | "
            f"{n_boxes / max(n_tiles, 1):.1f} boxes/tile"
        )

    # Absolute `path` root: Ultralytics resolves the Roboflow-style `../train/images`
    # against its own datasets dir, not against this file, and silently misses.
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        "nc: 1\n"
        'names: ["SeaGrape"]\n'
    )
    print(f"\nWrote {OUT}/data.yaml (single class)")


if __name__ == "__main__":
    main()
