"""
Step 3: Train the stage-1 single-class sea grape detector on 640px tiles.

yolo11s rather than yolo11n: tiles hold ~6-8 densely packed grapes and the nano
backbone bottlenecks recall on small clustered objects. The extra cost is small
next to the ~63 tiles each full image already costs at inference.
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = BASE_DIR / "build" / "tiles" / "data.yaml"
PROJECT = BASE_DIR / "build" / "runs"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="stage1_yolo11s")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from its last.pt instead of restarting")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Data:   {DATA_YAML}")

    # Resuming loads last.pt, which carries the optimizer state and epoch counter;
    # starting from the pretrained weights again would throw that away.
    last = PROJECT / args.name / "weights" / "last.pt"
    if args.resume:
        if not last.exists():
            raise SystemExit(f"--resume needs {last}, which does not exist")
        print(f"Resuming from {last}")
    model = YOLO(str(last)) if args.resume else YOLO(args.model)

    model.train(
        resume=args.resume,
        data=str(DATA_YAML),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=8,
        project=str(PROJECT),
        name=args.name,
        exist_ok=True,

        # Colour jitter is safe here and buys robustness to underwater lighting:
        # stage 1 is single-class, so shifting hue cannot flip a maturity label.
        # (The stage-2 classifier is the opposite case -- see 06_train_classifier.py.)
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,

        # Grapes have no canonical orientation, so full rotation and both flips
        # are free extra data.
        degrees=90.0,
        fliplr=0.5,
        flipud=0.5,

        # Small translate keeps tiny objects from being shoved off the tile.
        translate=0.1,
        scale=0.3,
        shear=0.0,
        perspective=0.0,

        # Mosaic helps dense small-object learning; turn it off at the end so the
        # final epochs tune box regression on undistorted tiles.
        mosaic=0.5,
        close_mosaic=15,
        mixup=0.0,      # would overlay ghost grapes onto real ones
        erasing=0.0,    # would delete grapes without deleting their labels

        cos_lr=True,
        optimizer="AdamW",
        lr0=1e-3,
        plots=True,
    )

    best = PROJECT / args.name / "weights" / "best.pt"
    print(f"\nBest weights: {best}")


if __name__ == "__main__":
    main()
