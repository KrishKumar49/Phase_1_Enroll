from fastapi import FastAPI, Response
from pydantic import BaseModel

from enroll import enroll_employee
from recognize import recognize_employee
from vehicle_entry import start_live_vehicle_entry, VehicleEntryRequest

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

@app.post("/vehicle-entry")
def vehicle_entry(data: VehicleEntryRequest):

    start_live_vehicle_entry(
        source=data.source,
        frame_skip=5,
        max_frames=100,
        show_window=False,
        persist=True,
        employee_id=data.employeeId,
        visit_id=data.visitId,
        camera_id=data.cameraId,
    )
    
    return {
        "status": "success",
        "message": "Vehicle processing completed"
    }