import os
import tempfile

import cv2
import numpy as np
import requests

from enroll import get_face_app
from database import get_all_employee_embeddings


MATCH_THRESHOLD = 0.50


def _download_image(image_url, destination_path):
    try:
        response = requests.get(
            image_url,
            stream=True,
            timeout=30
        )

        response.raise_for_status()

    except requests.RequestException:
        return False

    with open(destination_path, "wb") as image_file:
        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):
            if chunk:
                image_file.write(chunk)

    return True


def recognize_employee(image_url):
    os.makedirs("temp", exist_ok=True)

    temp_image_path = tempfile.mktemp(
        suffix=".jpg",
        dir="temp"
    )

    try:
        model = get_face_app()

        if not _download_image(
            image_url,
            temp_image_path
        ):
            return {
                "status": "failed",
                "message": "Could not download image"
            }

        frame = cv2.imread(temp_image_path)

        if frame is None:
            return {
                "status": "failed",
                "message": "Could not read image"
            }

        faces = model.get(frame)

        if len(faces) == 0:
            return {
                "status": "failed",
                "message": "No face detected"
            }

        face = max(
            faces,
            key=lambda f:
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )

        current_embedding = face.normed_embedding

        database_embeddings = get_all_employee_embeddings()

        best_match_employee = None
        best_similarity = -1

        for row in database_embeddings:
            employee_id = row[0]
            stored_embedding = np.array(row[1])

            similarity = np.dot(
                current_embedding,
                stored_embedding
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_employee = employee_id

        if best_similarity < MATCH_THRESHOLD:
            return {
                "status": "success",
                "matched": False,
                "similarity": float(best_similarity)
            }

        return {
            "status": "success",
            "matched": True,
            "employeeId": best_match_employee,
            "similarity": float(best_similarity)
        }

    except Exception as error:
        return {
            "status": "failed",
            "message": str(error)
        }

    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            