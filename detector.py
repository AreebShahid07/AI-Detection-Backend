from ultralytics import YOLO

def load_model():
    return YOLO("yolov8n.pt")  

def detect_objects(model, image):
    results = model(image, stream=False)
    names = model.names
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
            })

    return detections
