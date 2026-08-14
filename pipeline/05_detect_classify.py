"""
Two-stage sea grape inference: tiled YOLO detection -> MobileNetV3 maturity call.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import nms as tv_nms
from torchvision.ops import roi_align
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_YOLO = BASE_DIR / "build" / "runs" / "stage1_yolo11s" / "weights" / "best.pt"
DEFAULT_CLS = BASE_DIR / "build" / "runs" / "stage2_classifier" / "classifier_mobilenet_v3.pth"

NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

COLORS = {
    "Darkening": (34, 126, 230),
    "Harvestable": (113, 204, 46),
    "Whitening": (255, 255, 255),
    "Uncertain": (128, 128, 128),
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tile_offsets(total, tile, stride):
    if total <= tile:
        return [0]
    offs = list(range(0, total - tile + 1, stride))
    if offs[-1] != total - tile:
        offs.append(total - tile)
    return offs


class TwoStageDetector:
    def __init__(self, yolo_path=DEFAULT_YOLO, classifier_path=DEFAULT_CLS,
                 conf=0.25, uncertain_below=0.6):
        self.device = get_device()
        self.conf = conf
        self.uncertain_below = uncertain_below

        self.yolo = YOLO(str(yolo_path))

        ckpt = torch.load(classifier_path, map_location=self.device)
        self.class_names = ckpt["class_names"]
        self.img_size = ckpt.get("img_size", 128)
        # Must match the padding used to build the crops in 02_make_crops.py.
        self.context_pad = ckpt.get("context_pad", 0.15)

        self.classifier = models.mobilenet_v3_small(weights=None)
        self.classifier.classifier[3] = nn.Linear(
            self.classifier.classifier[3].in_features, len(self.class_names)
        )
        self.classifier.load_state_dict(ckpt["model_state_dict"])
        self.classifier.to(self.device).eval()

        self.mean = NORM_MEAN.to(self.device)
        self.std = NORM_STD.to(self.device)

        with torch.no_grad():  # warm the graph so the first real image is not the slow one
            self.classifier(torch.zeros(1, 3, self.img_size, self.img_size, device=self.device))

    # ---------- stage 1 ----------

    def detect(self, img_bgr, tile=640, stride=512, iou=0.45, downscale=1):
        """Tiled detection over the full image. Returns xyxy boxes in original pixels."""
        H, W = img_bgr.shape[:2]
        if downscale > 1:
            img_bgr = cv2.resize(img_bgr, (W // downscale, H // downscale),
                                 interpolation=cv2.INTER_AREA)
        h, w = img_bgr.shape[:2]

        if w <= tile and h <= tile:
            res = self.yolo.predict(img_bgr, conf=self.conf, verbose=False,
                                    device=str(self.device), max_det=1000)[0]
            boxes = res.boxes.xyxy.cpu()
            scores = res.boxes.conf.cpu()
        else:
            patches, origins = [], []
            for oy in tile_offsets(h, tile, stride):
                for ox in tile_offsets(w, tile, stride):
                    patches.append(img_bgr[oy:oy + tile, ox:ox + tile])
                    origins.append((ox, oy))

            all_boxes, all_scores = [], []
            margin = 5
            for i in range(0, len(patches), 32):
                chunk = patches[i:i + 32]
                results = self.yolo.predict(chunk, conf=self.conf, verbose=False,
                                            device=str(self.device), max_det=1000)
                for res, (ox, oy) in zip(results, origins[i:i + 32]):
                    b = res.boxes.xyxy.cpu().numpy()
                    s = res.boxes.conf.cpu().numpy()
                    if len(b) == 0:
                        continue
                    # Drop boxes clipped by an interior tile edge; the overlap
                    # guarantees a neighbouring tile holds the same grape whole.
                    interior_left = ox != 0
                    interior_top = oy != 0
                    interior_right = ox + tile < w
                    interior_bottom = oy + tile < h
                    clipped = np.zeros(len(b), bool)
                    if interior_left:
                        clipped |= b[:, 0] <= margin
                    if interior_top:
                        clipped |= b[:, 1] <= margin
                    if interior_right:
                        clipped |= b[:, 2] >= tile - margin
                    if interior_bottom:
                        clipped |= b[:, 3] >= tile - margin
                    keep = ~clipped
                    b = b[keep]
                    if len(b) == 0:
                        continue
                    b[:, [0, 2]] += ox
                    b[:, [1, 3]] += oy
                    all_boxes.append(b)
                    all_scores.append(s[keep])

            if not all_boxes:
                return torch.zeros((0, 4)), torch.zeros((0,))
            boxes = torch.from_numpy(np.concatenate(all_boxes)).float()
            scores = torch.from_numpy(np.concatenate(all_scores)).float()
            keep = tv_nms(boxes, scores, iou)
            boxes, scores = boxes[keep], scores[keep]

        if downscale > 1:
            boxes = boxes * downscale
        return boxes, scores

    # ---------- stage 2 ----------

    @torch.no_grad()
    def classify(self, img_bgr, boxes, tta=True):
        """Crop every box on the GPU with roi_align and classify in one batch."""
        if len(boxes) == 0:
            return np.zeros((0,), int), np.zeros((0,))

        H, W = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div_(255)
        img_t = img_t.unsqueeze(0)

        b = boxes.to(self.device).clone()
        pad_x = (b[:, 2] - b[:, 0]) * self.context_pad
        pad_y = (b[:, 3] - b[:, 1]) * self.context_pad
        b[:, 0] = (b[:, 0] - pad_x).clamp(0, W - 1)
        b[:, 1] = (b[:, 1] - pad_y).clamp(0, H - 1)
        b[:, 2] = (b[:, 2] + pad_x).clamp(1, W)
        b[:, 3] = (b[:, 3] + pad_y).clamp(1, H)

        rois = torch.cat([torch.zeros(len(b), 1, device=self.device), b], dim=1)
        crops = roi_align(img_t, rois, output_size=(self.img_size, self.img_size),
                          spatial_scale=1.0, sampling_ratio=2, aligned=True)
        crops = (crops - self.mean) / self.std

        probs_all = torch.zeros(len(crops), len(self.class_names), device=self.device)
        for i in range(0, len(crops), 512):
            chunk = crops[i:i + 512]
            out = self.classifier(chunk).softmax(1)
            if tta:
                out = (out + self.classifier(torch.flip(chunk, dims=[3])).softmax(1)) / 2
            probs_all[i:i + len(chunk)] = out

        probs, idx = probs_all.max(1)
        return idx.cpu().numpy(), probs.cpu().numpy()

    # ---------- driver ----------

    def predict(self, image, output_path=None, tile=640, stride=512, iou=0.45,
                downscale=1, tta=True):
        img_bgr = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image.copy()
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image}")

        t0 = time.perf_counter()
        boxes, det_conf = self.detect(img_bgr, tile, stride, iou, downscale)
        t_det = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cls_idx, cls_prob = self.classify(img_bgr, boxes, tta=tta)
        t_cls = (time.perf_counter() - t0) * 1000

        detections, tally = [], {}
        for box, dc, ci, cp in zip(boxes.numpy(), det_conf.numpy(), cls_idx, cls_prob):
            # Below threshold the maturity call is unreliable, but the grape is
            # real -- keep it in the count rather than deleting it.
            label = self.class_names[ci] if cp >= self.uncertain_below else "Uncertain"
            tally[label] = tally.get(label, 0) + 1
            detections.append({
                "box": [float(v) for v in box],
                "class": label,
                "classifier_conf": float(cp),
                "detector_conf": float(dc),
            })

        total = t_det + t_cls
        print(f"  {len(detections):>4} grapes | " + " ".join(f"{k}={v}" for k, v in sorted(tally.items()))
              + f" | detect {t_det:.0f}ms cls {t_cls:.0f}ms "
                f"= {total:.0f}ms ({1000 / total:.2f} img/s)")

        annotated = None
        if output_path:
            annotated = self.draw(img_bgr, detections)
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), annotated)

        return {"detections": detections, "tally": tally,
                "timing_ms": {"detect": t_det, "classify": t_cls}}

    def draw(self, img_bgr, detections, box_thickness=None):
        """Annotate detections. box_thickness overrides the resolution-derived default."""
        out = img_bgr.copy()
        W = out.shape[1]

        # W/600 puts a ~80px grape in an 8px frame on a 4640px image, which stays
        # visible once the picture is scaled down for a report. The old W/1700
        # gave 2px, a hairline that vanished at any viewing size.
        thickness = box_thickness if box_thickness else max(2, int(W / 600))
        overlay = out.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS.get(d["class"], (0, 255, 0)), thickness)
        # Boxes are drawn at higher opacity than before; at this line weight a
        # heavy blend would smear hundreds of overlapping frames into a haze.
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

        # Summary panel beats per-box text at this density -- hundreds of labels
        # would cover the grapes they describe.
        lines = [f"Total: {len(detections)}"]
        tally = {}
        for d in detections:
            tally[d["class"]] = tally.get(d["class"], 0) + 1
        lines += [f"{k}: {v}" for k, v in sorted(tally.items())]

        scale = max(0.6, W / 2200)
        pad, lh = int(14 * scale), int(34 * scale)
        box_w = int(360 * scale)
        cv2.rectangle(out, (pad, pad), (pad + box_w, pad + lh * len(lines) + pad), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            key = line.split(":")[0]
            cv2.putText(out, line, (pad * 2, pad + lh * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7,
                        COLORS.get(key, (255, 255, 255)), max(1, int(2 * scale)))
        return out


def main():
    ap = argparse.ArgumentParser(description="Two-stage sea grape detector")
    ap.add_argument("--image", default=str(BASE_DIR / "example_image.jpg"))
    ap.add_argument("--output", default=str(BASE_DIR / "build" / "predictions"))
    ap.add_argument("--yolo", default=str(DEFAULT_YOLO))
    ap.add_argument("--classifier", default=str(DEFAULT_CLS))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--downscale", type=int, default=1, help="2 is ~4x fewer tiles")
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    det = TwoStageDetector(args.yolo, args.classifier, conf=args.conf)
    src, dst = Path(args.image), Path(args.output)

    images = sorted(list(src.glob("*.jpg")) + list(src.glob("*.png"))) if src.is_dir() else [src]
    if not images:
        print(f"No images found at {src}")
        return

    for p in images:
        print(f"{p.name}:")
        out = dst / p.name if (dst.suffix == "" or src.is_dir()) else dst
        det.predict(p, output_path=out, tile=args.tile, stride=args.stride, iou=args.iou,
                    downscale=args.downscale, tta=not args.no_tta)
    print(f"\nSaved to {dst}")


if __name__ == "__main__":
    main()
