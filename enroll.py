import os
import tempfile
import zipfile

import cv2
import numpy as np
import requests

from insightface.app import FaceAnalysis

from database import save_employee_embedding


face_app = None


MODEL_HOME = os.path.join(
    os.path.dirname(__file__),
    "insightface_models"
)

MODEL_ROOT = os.path.join(
    MODEL_HOME,
    "models"
)

MODEL_CANDIDATES = (
    "buffalo_m",
    "buffalo_l",
)

MODEL_PACK_URLS = {
    "buffalo_m":
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip",

    "buffalo_l":
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
}


def _model_pack_path(model_name):

    return os.path.join(
        MODEL_ROOT,
        model_name
    )


def _has_model_files(model_name):

    model_path = _model_pack_path(
        model_name
    )

    return (
        os.path.isdir(model_path)
        and
        any(
            file_name.endswith(".onnx")
            for file_name in os.listdir(model_path)
        )
    )


def _download_model_pack(model_name):

    url = MODEL_PACK_URLS[model_name]

    os.makedirs(
        MODEL_ROOT,
        exist_ok=True
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".zip"
    ) as temp_zip:

        temp_zip_path = temp_zip.name

    try:

        print(f"Downloading {model_name}...")

        response = requests.get(
            url,
            stream=True,
            timeout=120
        )

        response.raise_for_status()

        with open(temp_zip_path, "wb") as zip_file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    zip_file.write(chunk)

        print("Extracting model files...")

        with zipfile.ZipFile(temp_zip_path) as archive:

            archive.extractall(
                path=MODEL_ROOT
            )

    finally:

        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)


def _download_video(
    video_url,
    destination_path
):

    try:

        response = requests.get(
            video_url,
            stream=True,
            timeout=30
        )

        response.raise_for_status()

    except requests.RequestException:

        return False

    with open(destination_path, "wb") as video_file:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:
                video_file.write(chunk)

    return True


def _ensure_model_pack(model_name):

    if _has_model_files(model_name):
        return

    _download_model_pack(model_name)

    if not _has_model_files(model_name):

        raise RuntimeError(
            f"Downloaded model "
            f"{model_name} is incomplete"
        )


def get_face_app():

    global face_app

    if face_app is None:

        last_error = None

        for model_name in MODEL_CANDIDATES:

            try:

                _ensure_model_pack(
                    model_name
                )

                print(
                    f"Loading model: "
                    f"{model_name}"
                )

                face_app = FaceAnalysis(
                    name=model_name,
                    root=MODEL_HOME,
                    providers=[
                        'CPUExecutionProvider'
                    ]
                )

                face_app.prepare(
                    ctx_id=-1,
                    det_size=(320, 320)
                )

                print(
                    f"{model_name} loaded successfully"
                )

                break

            except Exception as error:

                last_error = error

                face_app = None

        if face_app is None:

            raise RuntimeError(
                f"Could not initialize "
                f"InsightFace models: "
                f"{last_error}"
            )

    return face_app


def enroll_employee(
    video_url,
    employee_id
):

    os.makedirs(
        "temp",
        exist_ok=True
    )

    os.makedirs(
        "embeddings",
        exist_ok=True
    )

    temp_video_path = (
        f"temp/{employee_id}.mp4"
    )

    try:

        model = get_face_app()

    except Exception as error:

        return {
            "status": "failed",
            "message":
            f"Could not initialize "
            f"face model: {error}"
        }

    print("Downloading video...")

    if not _download_video(
        video_url,
        temp_video_path
    ):

        return {
            "status": "failed",
            "message":
            "Could not download video"
        }

    video = None

    try:

        video = cv2.VideoCapture(
            temp_video_path
        )

        if not video.isOpened():

            return {
                "status": "failed",
                "message":
                "Could not open video"
            }

        collected_face_data = []

        frame_count = 0

        while True:

            ret, frame = video.read()

            if not ret:
                break

            current_frame = frame_count

            frame_count += 1

            # Skip frames
            if current_frame % 15 != 0:
                continue

            # Resize for speed
            frame = cv2.resize(
                frame,
                (640, 480)
            )

            faces = model.get(frame)

            print(
                f"Frame {current_frame}: "
                f"{len(faces)} faces detected"
            )

            if len(faces) == 0:
                continue

            # Largest face
            face = max(
                faces,
                key=lambda f:
                (
                    (f.bbox[2] - f.bbox[0])
                    *
                    (f.bbox[3] - f.bbox[1])
                )
            )

            x1, y1, x2, y2 = map(
                int,
                face.bbox
            )

            h, w, _ = frame.shape

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face_width = x2 - x1
            face_height = y2 - y1

            # Draw detection box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Enrollment",
                frame
            )

            cv2.waitKey(1)

            # Face quality checks
            if face.det_score < 0.7:
                print("Low detection confidence")
                continue

            if face_width < 80 or face_height < 80:
                print("Face too small")
                continue

            face_crop = frame[
                y1:y2,
                x1:x2
            ]

            if (
                face_crop.size == 0
                or
                face_crop.shape[0] == 0
                or
                face_crop.shape[1] == 0
            ):

                continue

            gray_face = cv2.cvtColor(
                face_crop,
                cv2.COLOR_BGR2GRAY
            )

            blur_score = cv2.Laplacian(
                gray_face,
                cv2.CV_64F
            ).var()

            print(
                f"Frame {current_frame}: "
                f"Blur Score = "
                f"{blur_score:.2f}"
            )

            if blur_score < 30:

                print("Blurred face skipped")

                continue

            # Eye distance check
            left_eye = face.kps[0]
            right_eye = face.kps[1]

            eye_distance = abs(
                right_eye[0]
                - left_eye[0]
            )

            eye_ratio = (
                eye_distance /
                face_width
            )

            if eye_ratio < 0.25:

                print("Side face skipped")

                continue

            # Embedding
            embedding = np.array(
                face.normed_embedding,
                dtype=np.float32
            )

            embedding = (
                embedding /
                np.linalg.norm(
                    embedding
                )
            )

            # Duplicate filtering
            is_duplicate = False

            for saved_data in collected_face_data:

                saved_embedding = saved_data[0]

                similarity = np.dot(
                    embedding,
                    saved_embedding
                )

                if similarity > 0.995:

                    is_duplicate = True

                    print(
                        "Duplicate face skipped"
                    )

                    break

            if is_duplicate:
                continue

            collected_face_data.append(
                (
                    embedding,
                    face.det_score,
                    blur_score,
                    eye_ratio,
                    current_frame
                )
            )

            print(
                f"Accepted sample "
                f"{len(collected_face_data)}"
            )

        cv2.destroyAllWindows()

        if len(collected_face_data) < 5:

            return {
                "status": "failed",
                "message":
                "Not enough quality samples"
            }

        # Average embeddings
        final_embedding = np.mean(
            [
                data[0]
                for data in collected_face_data
            ],
            axis=0
        )

        final_embedding = (
            final_embedding /
            np.linalg.norm(
                final_embedding
            )
        )

        all_embeddings = np.array(
            [
                data[0]
                for data in collected_face_data
            ],
            dtype=np.float32
        )

        # Save locally
        np.save(
            f'embeddings/{employee_id}_all.npy',
            all_embeddings
        )

        np.save(
            f'embeddings/{employee_id}_mean.npy',
            final_embedding
        )

        # Save to database
        save_employee_embedding(
            employee_id,
            final_embedding.astype(
                np.float32
            ),
            all_embeddings.astype(
                np.float32
            ),
            len(collected_face_data)
        )

        return {
            "status": "success",
            "employeeId": employee_id,
            "embeddingsCount":
            len(collected_face_data),
            "embeddingStored": True
        }

    except Exception as error:

        return {
            "status": "failed",
            "message": str(error)
        }

    finally:

        if video is not None:
            video.release()

        cv2.destroyAllWindows()

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)