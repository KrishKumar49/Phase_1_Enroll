import argparse
import cv2
import numpy as np
from fast_alpr import ALPR
try:
    from PIL import Image
except Exception:
    Image = None


FRAME_SKIP_DEFAULT = 5


def start_live_plate_recognition(source=0, frame_skip=FRAME_SKIP_DEFAULT, max_frames=None, save_annotated_every=0, show_window=True):
    
    alpr = ALPR()

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
            detections = alpr.predict(frame)
        except Exception as e:
            print(f"ALPR predict error: {e}")
            detections = []

        annotated = None
        if detections:
            print(f"Frame {processed} (raw {frame_count}): Detected {len(detections)} plates")
            for det in detections:
                ocr = getattr(det, 'ocr', None)
                txt = getattr(ocr, 'text', '') if ocr is not None else ''
                conf = getattr(ocr, 'confidence', 0) if ocr is not None else 0
                print(f"  - Text: {txt}, Confidence: {conf}")

            try:
                annotated = alpr.draw_predictions(frame.copy())
            except Exception:
                annotated = frame

        def _to_cv2_image(img):
            if isinstance(img, np.ndarray):
                return img
            if Image is not None and isinstance(img, Image.Image):
                arr = np.array(img)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                return arr
            if isinstance(img, (bytes, bytearray)):
                arr = np.frombuffer(img, dtype=np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    return decoded
            try:
                if hasattr(img, 'numpy'):
                    arr = img.numpy()
                    return np.asarray(arr)
                if hasattr(img, 'cpu') and hasattr(img, 'numpy'):
                    arr = img.cpu().numpy()
                    return np.asarray(arr)
            except Exception:
                pass
            return None

        converted = _to_cv2_image(annotated)
        if converted is None:
            print(f"Warning: annotated image is type {type(annotated)}, falling back to raw frame")
            annotated = frame
        else:
            annotated = converted

        def _extract_bbox(d):
            try:
                det_obj = getattr(d, 'detection', None) or d
                bb = getattr(det_obj, 'bounding_box', None) or getattr(det_obj, 'bbox', None) or getattr(det_obj, 'box', None)
                if bb is not None:
                    if hasattr(bb, 'x1') and hasattr(bb, 'y1') and hasattr(bb, 'x2') and hasattr(bb, 'y2'):
                        return int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
                    if isinstance(bb, (list, tuple)) and len(bb) >= 4:
                        x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
                        if x2 <= 1 and y2 <= 1:
                            return None
                        if x2 <= x1 or y2 <= y1:
                            return int(x1), int(y1), int(x1 + x2), int(y1 + y2)
                        return int(x1), int(y1), int(x2), int(y2)
                if hasattr(d, 'bbox'):
                    b = d.bbox
                    if isinstance(b, (list, tuple)) and len(b) >= 4:
                        return int(b[0]), int(b[1]), int(b[2]), int(b[3])
            except Exception:
                pass
            return None

        for det in detections:
            bbox = _extract_bbox(det)
            ocr = getattr(det, 'ocr', None)
            txt = getattr(ocr, 'text', '') if ocr is not None else ''
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                h, w = annotated.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text_draw = txt if txt else '---'
                cv2.putText(annotated, text_draw, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if annotated is None:
            annotated = frame

        if save_annotated_every and (processed % save_annotated_every == 0):
            out_name = f"frame_{processed}_annotated.jpg"
            cv2.imwrite(out_name, annotated)
            print(f"Saved annotated frame: {out_name}")

        if show_window:
            cv2.imshow("Plate Recognition", annotated)
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
    parser = argparse.ArgumentParser(description='Live plate recognition from camera or video file')
    parser.add_argument('--source', '-s', default='0', help='Camera index (0) or path to video file')
    parser.add_argument('--skip', '-k', type=int, default=FRAME_SKIP_DEFAULT, help='Process every Nth frame')
    parser.add_argument('--max', '-m', type=int, default=None, help='Max processed frames (optional)')
    parser.add_argument('--save-every', '-e', type=int, default=0, help='Save annotated frame every N processed frames (0 = disabled)')
    parser.add_argument('--no-window', action='store_true', help='Do not show OpenCV window')

    args = parser.parse_args()

    src = args.source
    try:
        src_val = int(src)
    except Exception:
        src_val = src

    start_live_plate_recognition(source=src_val, frame_skip=args.skip, max_frames=args.max, save_annotated_every=args.save_every, show_window=not args.no_window)