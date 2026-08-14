"""
Step 3: Cut the stage-2 classifier training set out of the full-resolution images.

Each ground-truth box becomes one crop, taken from the original 4640x3480 pixels
rather than from a tile, so no resampling happens before the classifier sees it.

Crops carry 15% context padding on each side. Maturity is partly a judgement
about a grape relative to its neighbours, and a tight crop throws that context
away. Inference must pad by the same 15% or the classifier sees a different
distribution than it trained on.

The under-annotated images (002/029/030) are *included* here. Under-annotation
means boxes are missing, not that the boxes present are wrong -- every labelled
box is still a real grape with a real maturity label, and the rare classes need
every sample they can get.

Output is a torchvision ImageFolder tree:
    build/crops/<split>/<ClassName>/<image>_<index>.jpg
"""

import shutil
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
SRC = BASE_DIR / "build" / "detect"
OUT = BASE_DIR / "build" / "crops"

SPLITS = ["train", "valid", "test"]
CLASS_NAMES = ["Darkening", "Harvestable", "Whitening"]

CONTEXT_PAD = 0.15  # fraction of box size added on every side
MIN_CROP_PX = 12    # below this the crop carries no usable colour/texture signal


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in SPLITS:
        for name in CLASS_NAMES:
            (OUT / split / name).mkdir(parents=True, exist_ok=True)

    grand = {name: 0 for name in CLASS_NAMES}

    for split in SPLITS:
        counts = {name: 0 for name in CLASS_NAMES}
        skipped = 0
        sizes = []

        for img_path in sorted((SRC / split / "images").glob("*")):
            label_path = SRC / split / "labels" / (img_path.stem + ".txt")
            lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
            if not lines:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ! unreadable: {img_path.name}")
                continue
            H, W = img.shape[:2]
            stem = img_path.name.split("_")[0]

            for i, line in enumerate(lines):
                cls, cx, cy, bw, bh = line.split()
                cls = int(cls)
                cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H

                pad_x, pad_y = bw * CONTEXT_PAD, bh * CONTEXT_PAD
                x1 = max(0, int(round(cx - bw / 2 - pad_x)))
                y1 = max(0, int(round(cy - bh / 2 - pad_y)))
                x2 = min(W, int(round(cx + bw / 2 + pad_x)))
                y2 = min(H, int(round(cy + bh / 2 + pad_y)))

                if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
                    skipped += 1
                    continue

                name = CLASS_NAMES[cls]
                cv2.imwrite(
                    str(OUT / split / name / f"{stem}_{i:04d}.jpg"),
                    img[y1:y2, x1:x2],
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                counts[name] += 1
                sizes.append((x2 - x1, y2 - y1))

        for name in CLASS_NAMES:
            grand[name] += counts[name]
        total = sum(counts.values())
        med_w = sorted(s[0] for s in sizes)[len(sizes) // 2] if sizes else 0
        med_h = sorted(s[1] for s in sizes)[len(sizes) // 2] if sizes else 0
        print(
            f"  {split:<6} {total:>5} crops | "
            + "  ".join(f"{n}={counts[n]}" for n in CLASS_NAMES)
            + f" | median {med_w}x{med_h}px | skipped {skipped}"
        )

    total = sum(grand.values())
    print(f"\n  TOTAL  {total:>5} crops | " + "  ".join(f"{n}={grand[n]}" for n in CLASS_NAMES))
    worst = max(grand.values()) / max(min(grand.values()), 1)
    print(f"  Imbalance ratio (max:min) = {worst:.1f}:1 -> weighted sampler required")


if __name__ == "__main__":
    main()
