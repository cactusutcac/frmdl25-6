import os
import shutil

# Paths
image_dir = "dataset_preprocess/IXMAS_preprocess/IXMAS_720"   # Folder with selected .jpg images
video_dir = "dataset_preprocess/IXMAS_preprocess/IXMAS_raw"   # Folder with all 180 .avi videos
output_video_dir = "IXMAS"  # Output folder for matching videos

os.makedirs(output_video_dir, exist_ok=True)

# Get list of base names from selected images (without .jpg)
image_basenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg")]

# Match and copy corresponding .avi videos
for base in image_basenames:
    video_filename = base + ".avi"
    video_path = os.path.join(video_dir, video_filename)
    output_path = os.path.join(output_video_dir, video_filename)
    if os.path.exists(video_path):
        shutil.copy2(video_path, output_path)
    else:
        print(f"Missing video: {video_filename}")

print("Finished copying selected videos!")
