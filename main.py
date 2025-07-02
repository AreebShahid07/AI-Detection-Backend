from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from detector import load_model, detect_objects
from describe_image import describe_image
import cv2
import numpy as np
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()

@app.get("/")
def root():
    return {"message": "YOLO model backend is live!"}

@app.post("/detect-frame")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detections = detect_objects(model, image)
    return {"detections": detections}


@app.post("/describe-object")
async def describe(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    description = describe_image(temp_file_path)

    os.remove(temp_file_path)  

    return {"description": description}


#uvicorn main:app --reload
