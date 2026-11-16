import csv
import cv2
import time
import numpy as np
import openvino as ov

# Step 1: Load the exported YOLO model
core = ov.Core()

model = core.read_model(
    "yolo11n_openvino_model/yolo11n.xml"
)

# Step 2: Create test input

# Load the saved camera image
image = cv2.imread("capture.jpg")

if image is None:
    raise FileNotFoundError("capture.jpg not found!")

# Resize to the model's input dimensions
image = cv2.resize(image, (640, 640))

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Change shape from HWC to CHW
image = image.transpose(2, 0, 1)

# Convert to float32 and normalise pixel values
image = image.astype(np.float32) / 255.0

# Add batch dimension
test_input = np.expand_dims(image, axis=0)

print("Input shape:", test_input.shape)


# Step 3: Define benchmark function
def benchmark(device):
    print(f"\nTesting {device}...")

    compiled_model = core.compile_model(model, device)

    # Warm up the model
    for _ in range(10):
        compiled_model([test_input])

    times = []

    # Run 100 inference tests
    for _ in range(100):
        start = time.perf_counter()

        compiled_model([test_input])

        end = time.perf_counter()

        times.append((end - start) * 1000)

    average_ms = np.mean(times)
    fps = 1000 / average_ms

    print(f"Average inference: {average_ms:.2f} ms")
    print(f"Estimated throughput: {fps:.2f} inferences/sec")

    return average_ms

# Step 4: Test both devices

# Repeat benchmarking 5 times
cpu_results = []
gpu_results = []

for run in range(5):
    print(f"\n===== Benchmark Run {run + 1}/5 =====")

    cpu_results.append(benchmark("CPU"))
    gpu_results.append(benchmark("GPU.0"))

# Calculate overall averages
cpu_time = np.mean(cpu_results)
gpu_time = np.mean(gpu_results)

print("\n===== FINAL AVERAGES =====")
print(f"CPU: {cpu_time:.2f} ms")
print(f"GPU: {gpu_time:.2f} ms")


# Step 5: Compare performance
speedup = cpu_time / gpu_time

print("\n--- RESULTS ---")
print(f"CPU: {cpu_time:.2f} ms")
print(f"GPU: {gpu_time:.2f} ms")
print(f"GPU speedup: {speedup:.2f}x")


# Save benchmark results
with open("benchmark_results.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Device",
        "Average Latency (ms)",
        "Inferences per Second"
    ])

    writer.writerow([
        "CPU",
        round(cpu_time, 2),
        round(1000 / cpu_time, 2)
    ])

    writer.writerow([
        "Intel UHD GPU",
        round(gpu_time, 2),
        round(1000 / gpu_time, 2)
    ])

print("Benchmark results saved to CSV!")
