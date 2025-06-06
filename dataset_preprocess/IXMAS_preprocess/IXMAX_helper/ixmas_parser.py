import os
import re
import json
import cv2
from collections import defaultdict

def parse_filename(fname):
    pattern = re.compile(
        r"(?P<subject_raw>[a-z]+)(?P<subject_num>\d)_(?P<repetition>\d{2})_(?P<action>[\w\-]+)_cam\d+_frames_\d+_\d+\.avi"
    )
    match = pattern.fullmatch(fname)
    if not match:
        raise ValueError(f"Invalid IXMAS filename: {fname}")
    return {
        "subject_name": match.group("subject_raw"),
        "subject_instance": int(match.group("subject_num")),
        "repetition": int(match.group("repetition")),
        "action": match.group("action"),
        "video_id": fname
    }

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def generate_ixmas_clips(video_dir):
    grouped = defaultdict(list)
    subject_map = {}
    subject_counter = 0
    output = []

    for fname in sorted(os.listdir(video_dir)):
        if not fname.endswith(".avi"):
            continue
        try:
            meta = parse_filename(fname)
        except ValueError:
            continue

        key = (meta["subject_name"], meta["action"])
        grouped[key].append(meta)

        if meta["subject_name"] not in subject_map:
            subject_map[meta["subject_name"]] = subject_counter
            subject_counter += 1

    for (subject_name, action), clips in grouped.items():
        if len(clips) != 6:
            raise ValueError(f"Expected 6 clips for ({subject_name}, {action}), got {len(clips)}")

        clips_sorted = sorted(clips, key=lambda x: (x["subject_instance"], x["repetition"]))
        for i, clip in enumerate(clips_sorted):
            split = (
                "train" if i < 3 else
                "val" if i == 3 else
                "test"
            )

            video_path = os.path.join(video_dir, clip["video_id"])
            num_frames = get_video_frame_count(video_path)

            output.append({
                "video_id": clip["video_id"],
                "clip_id": clip["video_id"].replace(".avi", ""),
                "label": clip["action"],
                "subject": subject_map[subject_name],
                "scenario": "",  # no scenario info in IXMAS
                "start_frame": 0,
                "end_frame": num_frames - 1,
                "split": split
            })

    return output

if __name__ == "__main__":
    input_dir = "IXMAS"  # folder with .avi videos
    output_json = "ixmas_clips.json"

    clips = generate_ixmas_clips(input_dir)
    with open(output_json, "w") as f:
        json.dump(clips, f, indent=2)

    print(f"Wrote {len(clips)} clips to {output_json}")
