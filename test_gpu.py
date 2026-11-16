
from ultralytics import YOLO
import cv2

# Step 1: Load the exported OpenVINO model
model = YOLO("yolo11n_openvino_model/")

# Step 2: Run inference on Intel UHD Graphics
results = model.predict(
    source="capture.jpg",
    device="intel:gpu",
    verbose=True
)

# Step 3: Draw object detection boxes
annotated_image = results[0].plot()

# Step 4: Display the result
cv2.imshow("Intel GPU - Object Detection", annotated_image)

# Step 5: Wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()

print("GPU object detection completed!")
