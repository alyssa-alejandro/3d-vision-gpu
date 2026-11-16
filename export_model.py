
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

print("Exporting YOLO to OpenVINO format...")

model.export(format="openvino", imgsz=640)

print("Export completed!")
