import os

import cv2
import numpy as np
import requests

from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider']
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

def enroll_employee(video_url, employee_id):

    os.makedirs("temp", exist_ok=True)

    temp_video_path = f"temp/{employee_id}.mp4"

    try:
        response = requests.get(video_url, timeout=30)
    except requests.RequestException:
        return {
            "status": "failed",
            "message": "Could not download video"
        }

    if response.status_code != 200:
        return {
            "status": "failed",
            "message": "Could not download video"
        }

    with open(temp_video_path, "wb") as f:
        f.write(response.content)

    video = None

    try:
        video = cv2.VideoCapture(temp_video_path)

        if not video.isOpened():
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

            return {
                "status": "failed",
                "message": "Could not open video"
            }

        collected_face_data = []

        frame_count = 0


        # output_faces_dir = f'faces/{employee_id}'

        # os.makedirs(
        #     output_faces_dir,
        #     exist_ok=True
        # )

        os.makedirs(
            'embeddings',
            exist_ok=True
        )

        while True:

            ret, frame = video.read()

            if not ret:
                break

            if frame_count % 15 != 0:
                frame_count += 1
                continue

            x1, y1, x2, y2 = 0, 0, 0, 0 


            faces = app.get(frame)

            if len(faces) > 0:
                face = max(
                    faces,
                    key=lambda f:
                    (f.bbox[2] - f.bbox[0]) *
                    (f.bbox[3] - f.bbox[1])
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

                face_crop = None
                blur_score = 0.0
                eye_ratio = 0.0
                embedding = None
                

                if face.det_score < 0.7: 
                    pass
                elif face_width < 80 or face_height < 80: 
                    pass
                else:
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                        pass
                    else:
                        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

                        blur_score = cv2.Laplacian(
                            gray_face,
                            cv2.CV_64F
                        ).var()

                        print(f"Frame {frame_count}: Blur Score: {blur_score:.2f}")

                        if blur_score < 50:
                            frame_count += 1
                            continue

                        left_eye = face.kps[0]
                        right_eye = face.kps[1]
                        eye_distance = abs(right_eye[0] - left_eye[0])

                        if face_width > 0: 
                            eye_ratio = eye_distance / face_width
                        

                        if eye_ratio < 0.25: 
                            pass
                        else:
                            embedding = face.normed_embedding
                            is_duplicate = False
                            for saved_data in collected_face_data:
                                saved_embedding = saved_data[0] 
                                similarity = np.dot(embedding, saved_embedding)
                                if similarity > 0.97: 
                                    is_duplicate = True
                                    break

                            if not is_duplicate:
                                collected_face_data.append((embedding, face_crop, face.det_score, blur_score, eye_ratio, frame_count))

                                # cv2.imwrite(f"{output_faces_dir}/{frame_count}.jpg", face_crop)

            frame_count += 1


        if len(collected_face_data) < 10:
            return {
                "status": "failed",
                "message": "Not enough quality samples"
            }


        final_embedding = np.mean(
            [data[0] for data in collected_face_data],
            axis=0
        )

        np.save(
            f'embeddings/{employee_id}.npy',
            final_embedding
        )



        return {
            "status": "success",
            "employeeId": employee_id,
            "embeddingsCount": 1,
            "savedEmbeddings": f"embeddings/{employee_id}.npy",
            "embedding": final_embedding.tolist(),
        }

    except Exception as e:
        return {
            "status": "failed",
            "message": str(e)
        }
    finally:
        if video is not None:
            video.release()

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)