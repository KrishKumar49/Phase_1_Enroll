from fastapi import FastAPI, Response
from pydantic import BaseModel

from enroll import enroll_employee
from recognize import recognize_employee

app = FastAPI()

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
def recognize(
    image_url: str
):
    return recognize_employee(image_url)