
import openvino as ov

core = ov.Core()

print("Available OpenVINO devices:\n")

for device in core.available_devices:
    name = core.get_property(device, "FULL_DEVICE_NAME")

    print(f"Device ID: {device}")
    print(f"Device Name: {name}")
    print("-------------------------")
