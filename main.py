from fastapi import FastAPI, Response
from pydantic import BaseModel

from enroll import enroll_employee
from recognize import recognize_face

from fastapi import UploadFile, File
import numpy as np
import cv2

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EnrollRequest(BaseModel):
    employeeId: str
    videoUrl: str

@app.get("/")
def home():
    return {
        "message": "Enrollment API Running"
    }


@app.head("/")
def home_head():
    return Response(status_code=200)

@app.post("/enroll")
def enroll(data: EnrollRequest):

    result = enroll_employee(
        data.videoUrl,
        data.employeeId
    )

    return result

@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    np_arr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    results = recognize_face(frame)

    return {
        "results": results
    }