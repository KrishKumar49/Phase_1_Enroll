import os
import sys

import cv2
import torch
import numpy as np

FASTREID_PATH = os.path.join(os.path.dirname(__file__), "fast-reid")
if FASTREID_PATH not in sys.path:
    sys.path.append(FASTREID_PATH)

from ultralytics import YOLO
import torchvision.transforms as T

try:
    from fastreid.config import get_cfg
    from fastreid.engine import DefaultPredictor
except Exception as exc:
    get_cfg = None
    DefaultPredictor = None
    print(f"Warning: FastReID is unavailable ({exc}); vehicle embeddings will be skipped")

detector = YOLO("yolov8n.pt")

cfg = None
predictor = None

if get_cfg is not None and DefaultPredictor is not None:
    cfg = get_cfg()
    cfg.merge_from_file(
        "fast-reid/configs/VehicleID/bagtricks_R50-ibn.yml"
    )

    cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(__file__), "vehicleid_bot_R50-ibn.pth")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.isfile(cfg.MODEL.WEIGHTS):
        predictor = DefaultPredictor(cfg)
    else:
        print(f"Warning: FastReID checkpoint not found at {cfg.MODEL.WEIGHTS}; vehicle embeddings will be skipped")

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256,128)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


def get_vehicle_embedding(crop):

    if predictor is None:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    tensor = transform(crop_rgb)
    tensor = tensor.unsqueeze(0).to(cfg.MODEL.DEVICE)

    with torch.no_grad():
        embedding = predictor.model(
            {"images": tensor}
        )

    embedding = embedding.cpu().numpy().flatten()

    embedding = embedding / np.linalg.norm(embedding)

    return embedding


def get_vehicle_from_frame(frame):

    results = detector(frame, verbose=False)[0]

    for box in results.boxes:

        cls = int(box.cls[0])

        if cls not in [2,3,5,7]:
            continue

        x1,y1,x2,y2 = map(
            int,
            box.xyxy[0]
        )

        crop = frame[y1:y2, x1:x2]

        embedding = get_vehicle_embedding(crop)

        if embedding is not None:
            print(f"Embedding generated: {embedding.shape}")

        return {
            "vehicle_class": results.names[cls],
            "vehicle_embedding": embedding
        }


def start_live_vehicle_entry(source=0, frame_skip=5, max_frames=None, show_window=True):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print(f"Processing source: {source}")
    frame_count = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or cannot read frame")
            break

        frame_count += 1

        if frame_skip and (frame_count % frame_skip) != 0:
            continue

        processed += 1

        try:
            vehicle = get_vehicle_from_frame(frame)
        except Exception as e:
            print(f"Vehicle detection error: {e}")
            vehicle = None

        annotated = frame.copy()
        if vehicle is not None:
            # draw a simple notification on the frame
            cv2.putText(annotated, f"{vehicle['vehicle_class']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            print(f"Detected vehicle: {vehicle['vehicle_class']}")

        if show_window:
            cv2.imshow("Vehicle Entry", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested exit")
                break

        if max_frames and processed >= max_frames:
            print(f"Reached max processed frames: {max_frames}")
            break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # simple CLI for manual runs
    import argparse

    parser = argparse.ArgumentParser(description='Live vehicle entry detection from camera or video file')
    parser.add_argument('--source', '-s', default='0', help='Camera index (0) or path to video file')
    parser.add_argument('--skip', '-k', type=int, default=5, help='Process every Nth frame')
    parser.add_argument('--max', '-m', type=int, default=None, help='Max processed frames (optional)')
    parser.add_argument('--no-window', action='store_true', help='Do not show OpenCV window')

    args = parser.parse_args()

    src = args.source
    try:
        src_val = int(src)
    except Exception:
        src_val = src

    start_live_vehicle_entry(source=src_val, frame_skip=args.skip, max_frames=args.max, show_window=not args.no_window)