from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

IMAGE_NAME = "4df43a1e-3377112912_10AUG25_1458_00.png"

TRAINED_WEIGHTS = ROOT / "runs" / "train" / "deadbody_v1" / "weights" / "best.pt"
FALLBACK_WEIGHTS = ROOT / "yolo11n.pt"

OUTPUT_DIR = ROOT / "outputs"

CONF_THRES = 0.20
IMGSZ = 1280


def resolve_image_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    in_images = ROOT / "images" / name_or_path
    if in_images.exists():
        return in_images
    raise FileNotFoundError(
        f"Image not found: '{name_or_path}'\n"
        f"  Checked:\n"
        f"   - {p.resolve()}\n"
        f"   - {in_images.resolve()}"
    )


def find_latest_trained_weights(train_dir: Path) -> Path | None:
    if not train_dir.exists():
        return None
    exp_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    if not exp_dirs:
        return None
    exp_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    for exp in exp_dirs:
        best_pt = exp / "weights" / "best.pt"
        if best_pt.exists():
            return best_pt
    return None


def main() -> None:
    img_path = resolve_image_path(IMAGE_NAME)
    print(f"  Target Image : {img_path.name}")
    print(f"  Full Path    : {img_path}")

    latest_weights = find_latest_trained_weights(ROOT / "runs" / "train")
    if latest_weights:
        model_path = latest_weights
        print(f"  Model Weights: Latest trained model ({model_path.relative_to(ROOT)})")
    elif FALLBACK_WEIGHTS.exists():
        model_path = FALLBACK_WEIGHTS
        print(f"  Model Weights: Pretrained baseline ({model_path.name})")
    else:
        raise FileNotFoundError("No model weights found.")

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Compute      : Device {device} ({device_name})")

    model = YOLO(str(model_path))

    print("\n── Running Inference ───────────────────────────────────────────")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(img_path),
        imgsz=IMGSZ,
        conf=CONF_THRES,
        iou=0.45,
        device=device,
        save=False,
    )

    for r in results:
        boxes = r.boxes
        print(f"  Detections found: {len(boxes)}")

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            print(f"   [#%d] Class: '%s' | Conf: %.2f | Box: [%d, %d, %d, %d]" % (i + 1, cls_name, conf, x1, y1, x2, y2))

        annotated_img = r.plot()
        out_file = OUTPUT_DIR / f"marked_{img_path.stem}.png"

        import cv2
        cv2.imwrite(str(out_file), annotated_img)

        print(f"\n  [OK] Marked image saved to -> {out_file}")


if __name__ == "__main__":
    main()
