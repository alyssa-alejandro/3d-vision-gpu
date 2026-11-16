
import cv2
import time
from ultralytics import YOLO

# Step 1: Load the pretrained YOLO model
model = YOLO("yolo11n_openvino_model/")
print("YOLO device:", model.device)

# Step 2: Open webcam
camera = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not camera.isOpened():
    print("Failed to connect to camera.")
    exit()

# Step 3: Initialise timer
previous_time = time.perf_counter()

# Step 4: Start camera loop
while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to read frame.")
        break

    # Step 5: Run object detection
    results = model.predict(
    source=frame,
    device="intel:gpu",
    verbose=False
    )

    # Step 6: Draw bounding boxes
    annotated_frame = results[0].plot()

    # Step 7: Calculate FPS
    current_time = time.perf_counter()

    elapsed_time = current_time - previous_time

    fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0

    previous_time = current_time

    
    # Display FPS on the camera
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # Step 8: Display camera
    cv2.imshow("3D Vision - Live Object Detection", annotated_frame)

    # Step 9: Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Step 10: Release resources
camera.release()
cv2.destroyAllWindows()
