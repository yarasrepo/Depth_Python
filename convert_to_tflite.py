import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("yolov8n_tf")

# Add this line 👇 to allow TF ops fallback
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable built-in TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS     # enable TF ops fallback (like SplitV)
]

# Optional optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert and save
tflite_model = converter.convert()

with open("yolov8n.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ YOLOv8 model converted successfully with TF Select fallback")
