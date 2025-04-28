# import logging
# from fastapi import FastAPI, UploadFile, File, HTTPException
# import uvicorn
# import numpy as np
# import torch
# from PIL import Image, ImageDraw, ImageFont
# from io import BytesIO
# import torchvision.transforms as transforms
# from fastapi.responses import JSONResponse
# import tensorflow as tf

# # Set up logging
# logging.basicConfig(level=logging.INFO)

# app = FastAPI()

# # Load MiDaS Model
# model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
# model.eval()

# # Load TensorFlow Lite model for object detection
# TFLITE_MODEL_PATH = "mobilenetv1.tflite"
# interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Image transformation for MiDaS
# depth_transform = transforms.Compose([
#     transforms.Resize((384, 384)),
#     transforms.ToTensor()
# ])

# # i change it to labels.txt later on
# COCO_LABELS = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#     'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#     'hair drier', 'toothbrush'
# ]


# # for bounding box visualization
# def draw_boxes(image, detections):
#     draw = ImageDraw.Draw(image)
#     try:
#         font = ImageFont.truetype("arial.ttf", size=12)
#     except:
#         font = ImageFont.load_default()
    
#     for obj in detections:
#         box = obj["box"]
#         label = f'{obj["label"]} ({obj["confidence"]:.2f})'
#         draw.rectangle(box, outline="red", width=2)
#         draw.text((box[0], box[1] - 10), label, fill="red", font=font)

#     return image


# # Object detection function
# def detect_object(image):
#     input_shape = input_details[0]['shape'][1:3]
#     resized_image = image.resize(input_shape)
#     input_data = np.expand_dims(resized_image, axis=0)

#     if input_details[0]['dtype'] == np.uint8:
#         input_data = input_data.astype(np.uint8)
#     else:
#         input_data = (input_data / 255.0).astype(np.float32)

#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()

#     boxes = interpreter.get_tensor(output_details[0]['index'])[0]
#     classes = interpreter.get_tensor(output_details[1]['index'])[0]
#     scores = interpreter.get_tensor(output_details[2]['index'])[0]

#     detected_objects = []
#     for i in range(len(scores)):
#         if scores[i] > 0.5:  # Confidence threshold
#             class_id = int(classes[i])
#             label = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else "unknown"
#             ymin, xmin, ymax, xmax = boxes[i]
#             width, height = image.size
#             box = [
#                 int(xmin * width),
#                 int(ymin * height),
#                 int(xmax * width),
#                 int(ymax * height)
#             ]
#             detected_objects.append({
#                 "label": label,
#                 "confidence": float(scores[i]),
#                 "box": box
#             })

#     return detected_objects


# @app.get("/")
# async def hello():
#     return {"message": "Hello, world!"}

# @app.post("/predict")
# async def predict_depth(file: UploadFile = File(...)):
#     try:
#         logging.info(f"Received file: {file.filename}")
#         image = Image.open(BytesIO(await file.read())).convert("RGB")
        
#         # Depth estimation
#         image_tensor = depth_transform(image).unsqueeze(0)
#         with torch.no_grad():
#             depth_map = model(image_tensor)
#         depth_map = depth_map.squeeze().cpu().numpy()
#         avg_depth = np.mean(depth_map)
        
#         # Dynamic calibration factor
#         calibration_factor = 1.0 / avg_depth
#         height, width = depth_map.shape
#         center_pixel_depth = depth_map[height // 2, width // 2]
#         real_distance = calibration_factor * center_pixel_depth
        
#         # Object detection
#         detected_objects = detect_object(image)
        
#         logging.info(f"Estimated distance: {real_distance:.2f}m")
#         if detected_objects:
#             logging.info(f"Detected objects: {detected_objects}")
#         else:
#             logging.info("No objects detected.")
        
#         return JSONResponse(content={
#             "estimated_distance_meters": float(real_distance),
#             "detected_objects": detected_objects
#         })
    
#     except Exception as e:
#         logging.error(f"Error: {e}")
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
