import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
from fastapi.responses import JSONResponse

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Load MiDaS Model
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize to MiDaS input size
    transforms.ToTensor()
])

@app.get("/")
async def hello():
    return {"message": "Hello, world!"}

@app.post("/predict")
async def predict_depth(file: UploadFile = File(...)):
    try:
        logging.info(f"Received file: {file.filename}")

        # Read and process the image
        image = Image.open(BytesIO(await file.read())).convert("RGB")

        # Transform the image
        image_tensor = transform(image).unsqueeze(0)

        # Generate the depth map using MiDaS
        with torch.no_grad():
            depth_map = model(image_tensor)

        # Convert depth map to numpy array
        depth_map = depth_map.squeeze().cpu().numpy()

        # Calculate the average depth value across the image
        avg_depth = np.mean(depth_map)

      

        # Heuristic calibration factor: Adjust based on the typical depth value
        calibration_factor = 1.0 / avg_depth

        # Calculate the depth at the center pixel
        height, width = depth_map.shape
        center_pixel_depth = depth_map[height // 2, width // 2]

        # Convert depth to real-world distance using the dynamic calibration factor
        real_distance = calibration_factor * center_pixel_depth

        logging.info(f"Estimated real-world distance: {real_distance:.2f} meters")

        return JSONResponse(content={"estimated_distance_meters": float(real_distance)})

    except Exception as e:
        logging.error(f"Error processing the image: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
