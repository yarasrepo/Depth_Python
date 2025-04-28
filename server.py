import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
import mediapipe as mp

logging.basicConfig(level=logging.INFO)
app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load MiDaS (depth estimation) ===
midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_model = midas_model.to(device)
midas_model.eval()
depth_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

# === Load MediaPipe Object Detector ===
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_detector = ObjectDetector.create_from_options(
    ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path="efficientdet_lite0.tflite"),
        running_mode=VisionRunningMode.IMAGE,
        score_threshold=0.5,
        max_results=5
    )
)

# === Helper: Convert PIL to ndarray ===
def pil_to_ndarray(image: Image.Image):
    return np.array(image)

@app.get("/")
async def hello():
    return {"message": "Hybrid MiDaS + MediaPipe backend is live!"}

@app.post("/predict")
async def predict_depth(file: UploadFile = File(...)):
    try:
        logging.info(f"Received file: {file.filename}")
        image = Image.open(BytesIO(await file.read())).convert("RGB")

        # === Depth Estimation ===
        image_tensor = depth_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            depth_map = midas_model(image_tensor)
        depth_map = depth_map.squeeze().cpu().numpy()
        avg_depth = np.mean(depth_map)

        calibration_factor = 1.0 / avg_depth
        h, w = depth_map.shape
        center_pixel_depth = depth_map[h // 2, w // 2]
        real_distance = calibration_factor * center_pixel_depth
        logging.info(f"Real distance: {center_pixel_depth:.2f} cm")

        if center_pixel_depth < 1500:
        # === Object Detection ===
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=pil_to_ndarray(image))
            results = mp_detector.detect(mp_image)

            detected_objects = []
            center_x, center_y = image.width // 2, image.height // 2
            object_in_front = None

            if results.detections:
                logging.info("Detected objects:")
                for det in results.detections:
                    category = det.categories[0]
                    bbox = det.bounding_box
                    x1, y1 = bbox.origin_x, bbox.origin_y
                    x2, y2 = x1 + bbox.width, y1 + bbox.height

                    logging.info(f"→ {category.category_name} ({category.score:.2f})")

                    if (x1 <= center_x <= x2) and (y1 <= center_y <= y2) and (category.score > 0.6):
                        object_in_front = {
                            "label": category.category_name,
                            "confidence": float(category.score),
                            "box": [int(x1), int(y1), int(x2), int(y2)]
                        }

                    detected_objects.append({
                        "label": category.category_name,
                        "confidence": float(category.score),
                        "box": [int(x1), int(y1), int(x2), int(y2)]
                    })
                if object_in_front:
                    alert = {
                        "alert_type": "object",
                        "label": object_in_front["label"],
                        "distance": float(center_pixel_depth)
                    }
                else:
                    alert = {
                        "alert_type": "object",
                        "distance": float(center_pixel_depth)
                    }
            else:
                logging.info("No objects detected.")
                alert = {
                    "alert_type": "surface",
                    "distance": float(center_pixel_depth)
                }

        else:
            alert = {
                "alert_type": "none"
            }

        logging.info(f"Final alert: {alert}")
        return JSONResponse(content=alert)

    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
