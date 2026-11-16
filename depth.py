
from transformers import pipeline
from PIL import Image

# Step 1: Load your saved camera image
image = Image.open("capture.jpg").convert("RGB")

# Step 2: Load the pretrained depth model
print("Loading Depth Anything V2...")

depth_model = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device="cpu"
)

# Step 3: Estimate depth
print("Estimating depth...")

result = depth_model(image)


# Get the model's raw depth predictions
raw_depth = result["predicted_depth"]

# Convert the tensor to a NumPy array
raw_depth = raw_depth.squeeze().cpu().numpy()

print("\n--- RAW DEPTH PREDICTIONS ---")
print("Shape:", raw_depth.shape)
print("Minimum:", raw_depth.min())
print("Maximum:", raw_depth.max())

height, width = raw_depth.shape

print(
    "Centre depth:",
    raw_depth[height // 2, width // 2]
)


# Step 4: Save the depth map
depth_map = result["depth"]


import numpy as np

# Convert the depth image to a NumPy array
depth_array = np.array(depth_map)

print("Depth map shape:", depth_array.shape)
print("Minimum value:", depth_array.min())
print("Maximum value:", depth_array.max())

# Inspect the centre pixel
height, width = depth_array.shape

centre_depth = depth_array[height // 2, width // 2]

print("Centre pixel depth value:", centre_depth)

depth_map.save("depth_map.png")

print("Depth map saved successfully!")

# Step 5: Display the depth map
depth_map.show()
