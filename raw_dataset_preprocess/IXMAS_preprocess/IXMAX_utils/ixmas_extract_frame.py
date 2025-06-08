import os
import cv2

# Paths
source_dir = "dataset_preprocess/IXMAS_preprocess/IXMAS_raw"        # Folder with 3600 .avi videos
output_dir = "dataset_preprocess/IXMAS_preprocess/IXMAS_img"    # Folder to save extracted images
os.makedirs(output_dir, exist_ok=True)

# Process each .avi file
for filename in os.listdir(source_dir):
    if filename.endswith(".avi"):
        video_path = os.path.join(source_dir, filename)
        image_name = os.path.splitext(filename)[0] + ".jpg"
        image_path = os.path.join(output_dir, image_name)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {filename}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = total_frames // 2  # Middle frame for visual balance

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()

        if ret:
            cv2.imwrite(image_path, frame)
        else:
            print(f"Failed to read frame from: {filename}")

        cap.release()

print("Images saved in:", output_dir)
