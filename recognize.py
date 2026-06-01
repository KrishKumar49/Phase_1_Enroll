import ast

import cv2
import numpy as np

from enroll import get_face_app
from database import get_all_employee_embeddings


MATCH_THRESHOLD = 0.35
FRAME_SKIP = 5

def _load_known_embeddings():

    database_rows = get_all_employee_embeddings()

    known_embeddings = {}

    for employee_id, embedding in database_rows:

        if isinstance(embedding, (str, np.str_)):
            embedding = ast.literal_eval(embedding)

        embedding = np.array(
            embedding,
            dtype=np.float32
        )

        norm = np.linalg.norm(embedding)

        if norm > 0:
            embedding = embedding / norm

        employee_embeddings = known_embeddings.setdefault(
            employee_id,
            []
        )

        employee_embeddings.append(embedding)

    return known_embeddings

print("Loading embeddings from database...")
KNOWN_EMBEDDINGS = _load_known_embeddings()

print("Loading InsightFace model...")
MODEL = get_face_app()

def recognize_face(frame):

    known_embeddings = KNOWN_EMBEDDINGS

    model = MODEL

    small_frame = cv2.resize(
        frame,
        (640, 480)
    )

    faces = model.get(small_frame)

    results = []

    for face in faces:

        x1, y1, x2, y2 = map(
            int,
            face.bbox
        )

        best_match_id = None
        best_match_score = -1.0

        embedding = np.array(
            face.normed_embedding,
            dtype=np.float32
        )

        for (
            employee_id,
            employee_embeddings
        ) in known_embeddings.items():

            for stored_embedding in employee_embeddings:

                similarity = np.dot(
                    embedding,
                    stored_embedding
                )

                if similarity > best_match_score:

                    best_match_score = similarity
                    best_match_id = employee_id

        if best_match_score > MATCH_THRESHOLD:

            results.append({
                "employee_id": best_match_id,
                "score": float(best_match_score),
                "bbox": [x1, y1, x2, y2]
            })

        else:

            results.append({
                "employee_id": "Unknown",
                "score": float(best_match_score),
                "bbox": [x1, y1, x2, y2]
            })

    return results

def start_live_recognition(camera_source=0):

    print("Loading embeddings from database")

    known_embeddings = KNOWN_EMBEDDINGS
    model = MODEL

    print("Starting video capture")

    total_embeddings = sum(
        len(employee_embeddings)
        for employee_embeddings in known_embeddings.values()
    )

    print(
        f"Loaded {len(known_embeddings)} employees "
        f"with {total_embeddings} embeddings"
    )

    video = cv2.VideoCapture(camera_source)

    if not video.isOpened():
        print("Could not open video source")
        return

    frame_count = 0

    last_seen = {}

    # Reuse detections between skipped frames
    last_faces = []

    while True:

        ret, frame = video.read()

        if not ret:
            print("Could not read frame")
            break

        frame_count += 1

        small_frame = cv2.resize(
            frame,
            (640, 480)
        )

        # Run face detection every FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:

            last_faces = model.get(small_frame)

            print(
                "Faces detected:",
                len(last_faces)
            )

        faces = last_faces

        for face in faces:

            x1, y1, x2, y2 = map(
                int,
                face.bbox
            )

            best_match_id = None
            best_match_score = -1.0

            # InsightFace already provides normalized embeddings
            embedding = np.array(
                face.normed_embedding,
                dtype=np.float32
            )

            # Compare against database embeddings
            for (
                employee_id,
                employee_embeddings
            ) in known_embeddings.items():

                for stored_embedding in employee_embeddings:

                    similarity = np.dot(
                        embedding,
                        stored_embedding
                    )

                    if similarity > best_match_score:

                        best_match_score = similarity
                        best_match_id = employee_id

            print(
                "Best Match:",
                best_match_id,
                "Score:",
                f"{best_match_score:.2f}"
            )

            if best_match_score > MATCH_THRESHOLD:

                cv2.rectangle(
                    small_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    small_frame,
                    f"{best_match_id} "
                    f"({best_match_score:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                current_time = (
                    cv2.getTickCount()
                    / cv2.getTickFrequency()
                )

                if (
                    best_match_id not in last_seen
                    or
                    current_time
                    - last_seen[best_match_id]
                    > 1
                ):

                    last_seen[best_match_id] = current_time

                    print(
                        f"Recognized employee "
                        f"{best_match_id} "
                        f"with score "
                        f"{best_match_score:.2f}"
                    )

            else:

                cv2.rectangle(
                    small_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    small_frame,
                    "Unknown",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

        # Cleanup old entries
        current_time = (
            cv2.getTickCount()
            / cv2.getTickFrequency()
        )

        to_remove = []

        for (
            employee_id,
            last_seen_time
        ) in last_seen.items():

            if (
                current_time
                - last_seen_time
            ) > 5:

                to_remove.append(employee_id)

        for employee_id in to_remove:
            del last_seen[employee_id]

        # Always show frame
        cv2.imshow(
            "Live Recognition",
            small_frame
        )

        if (
            cv2.waitKey(1)
            & 0xFF
            == ord('q')
        ):
            break

    video.release()

    cv2.destroyAllWindows()