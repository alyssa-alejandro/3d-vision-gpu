
import csv
import matplotlib.pyplot as plt

# Read results from CSV
devices = []
latencies = []

with open("benchmark_results.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        devices.append(row["Device"])
        latencies.append(float(row["Average Latency (ms)"]))

# Create graph
plt.figure(figsize=(8, 5))

bars = plt.bar(devices, latencies)

plt.title("YOLO11 Nano - CPU vs GPU Inference")
plt.ylabel("Average Inference Latency (ms)")
plt.xlabel("Device")

# Display values above each bar
for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f} ms",
        ha="center",
        va="bottom"
    )

plt.ylim(0, max(latencies) * 1.2)
plt.tight_layout()

# Save graph
plt.savefig("benchmark_graph.png", dpi=150)

# Show graph
plt.show()
