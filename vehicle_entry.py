import os
import sys
import logging
import importlib

import cv2
import torch
import numpy as np

BASE_DIR = os.path.dirname(__file__)
from download_models import ensure_vehicle_model_assets

ensure_vehicle_model_assets()

FASTREID_PATH = os.getenv("FASTREID_PATH", os.path.join(BASE_DIR, "fast-reid"))
if FASTREID_PATH and os.path.isdir(FASTREID_PATH) and FASTREID_PATH not in sys.path:
    sys.path.append(FASTREID_PATH)

from ultralytics import YOLO
import torchvision.transforms as T

from database import get_active_visit_id, save_vehicle_entry_record

try:
    fastreid_config = importlib.import_module("fastreid.config")
    fastreid_engine = importlib.import_module("fastreid.engine")
    get_cfg = fastreid_config.get_cfg
    DefaultPredictor = fastreid_engine.DefaultPredictor
except Exception as exc:
    get_cfg = None
    DefaultPredictor = None
    print(f"Warning: FastReID is unavailable ({exc}); vehicle embeddings will be skipped")

logger = logging.getLogger(__name__)

DETECTION_MODEL_PATH = os.getenv("VEHICLE_DETECTION_MODEL", os.path.join(BASE_DIR, "yolov8n.pt"))
REID_WEIGHTS_PATH = os.getenv("VEHICLE_REID_WEIGHTS", os.path.join(BASE_DIR, "vehicleid_bot_R50-ibn.pth"))
REID_CONFIG_PATH = os.getenv(
    "VEHICLE_REID_CONFIG",
    os.path.join(FASTREID_PATH, "configs", "VehicleID", "bagtricks_R50-ibn.yml"),
)

if not os.path.isdir(FASTREID_PATH) or not any(os.scandir(FASTREID_PATH)):
    print(f"Warning: FastReID source tree is missing or empty at {FASTREID_PATH}; embeddings will be skipped")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

if not os.path.isfile(DETECTION_MODEL_PATH):
    raise FileNotFoundError(f"Vehicle detection model not found: {DETECTION_MODEL_PATH}")

detector = YOLO(DETECTION_MODEL_PATH)

cfg = None
predictor = None

if get_cfg is not None and DefaultPredictor is not None:
    cfg = get_cfg()
    if os.path.isfile(REID_CONFIG_PATH):
        cfg.merge_from_file(REID_CONFIG_PATH)

        cfg.MODEL.WEIGHTS = REID_WEIGHTS_PATH
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        if os.path.isfile(cfg.MODEL.WEIGHTS):
            predictor = DefaultPredictor(cfg)
        else:
            logger.warning("FastReID checkpoint not found at %s; vehicle embeddings will be skipped", cfg.MODEL.WEIGHTS)
    else:
        logger.warning("FastReID config not found at %s; vehicle embeddings will be skipped", REID_CONFIG_PATH)

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

    if crop is None or crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    tensor = transform(crop_rgb)
    tensor = tensor.unsqueeze(0).to(cfg.MODEL.DEVICE)

    with torch.no_grad():
        embedding = predictor.model({"images": tensor})

    embedding = embedding.cpu().numpy().flatten()

    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None

    embedding = embedding / norm

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


def _resolve_visit_id(employee_id, visit_id):
    if visit_id:
        return visit_id

    if employee_id:
        return get_active_visit_id(employee_id)

    return None


def start_live_vehicle_entry(
    source=0,
    frame_skip=5,
    max_frames=None,
    show_window=False,
    persist=False,
    employee_id=None,
    visit_id=None,
    camera_id=None,
):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Could not open video source %s", source)
        return

    logger.info("Processing source: %s", source)
    resolved_visit_id = _resolve_visit_id(employee_id, visit_id)
    if persist and not resolved_visit_id:
        logger.warning(
            "Persistence is enabled but no visit_id could be resolved. Provide visit_id or employee_id."
        )

    frame_count = 0
    processed = 0
    saved_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of stream or cannot read frame")
            break

        frame_count += 1

        if frame_skip and (frame_count % frame_skip) != 0:
            continue

        processed += 1

        try:
            vehicle = get_vehicle_from_frame(frame)
        except Exception as e:
            logger.exception("Vehicle detection error: %s", e)
            vehicle = None

        annotated = frame.copy()
        if vehicle is not None:
            cv2.putText(annotated, f"{vehicle['vehicle_class']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            logger.info("Detected vehicle: %s", vehicle["vehicle_class"])

            if persist and resolved_visit_id and employee_id:
                try:
                    entry_id = save_vehicle_entry_record(
                        employee_id=employee_id,
                        visit_id=resolved_visit_id,
                        vehicle_embedding=vehicle.get("vehicle_embedding"),
                        vehicle_class=vehicle.get("vehicle_class"),
                        plate_number=None,
                        camera_id=camera_id,
                    )
                    saved_detections += 1
                    logger.info("Saved entry record id=%s", entry_id)
                except Exception as exc:
                    logger.exception("Failed to save vehicle entry record: %s", exc)

        if show_window:
            cv2.imshow("Vehicle Entry", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested exit")
                break

        if max_frames and processed >= max_frames:
            logger.info("Reached max processed frames: %s", max_frames)
            break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    logger.info("Vehicle entry run complete: processed=%s saved=%s", processed, saved_detections)


if __name__ == '__main__':
    # simple CLI for manual runs
    import argparse

    parser = argparse.ArgumentParser(description='Live vehicle entry detection from camera or video file')
    parser.add_argument('--source', '-s', default='0', help='Camera index (0) or path to video file')
    parser.add_argument('--skip', '-k', type=int, default=5, help='Process every Nth frame')
    parser.add_argument('--max', '-m', type=int, default=None, help='Max processed frames (optional)')
    parser.add_argument('--show-window', action='store_true', help='Show OpenCV window')
    parser.add_argument('--persist', action='store_true', help='Save detections to entry_records')
    parser.add_argument('--employee-id', default=None, help='Employee ID used to resolve the active visit')
    parser.add_argument('--visit-id', default=None, help='Explicit visit_id for entry_records')
    parser.add_argument('--camera-id', default=None, help='Camera ID to save with entry_records')

    args = parser.parse_args()

    src = args.source
    try:
        src_val = int(src)
    except Exception:
        src_val = src

    start_live_vehicle_entry(
        source=src_val,
        frame_skip=args.skip,
        max_frames=args.max,
        show_window=args.show_window,
        persist=args.persist,
        employee_id=args.employee_id,
        visit_id=args.visit_id,
        camera_id=args.camera_id,
    )