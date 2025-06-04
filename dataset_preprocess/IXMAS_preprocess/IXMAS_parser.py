import os
import re
import json
from collections import defaultdict

def parse_filename(fname):
    """
    Parse IXMAS filename into components. 
    Example: alba1_01_check-watch_cam2_frames_0053_0097.avi
    Returns: subject name (str), action (str), start_frame (int), end_frame (int)
    """
    pattern = re.compile(
        r"(?P<subject_raw>[a-z]+)(?P<subject_num>\d)_(?P<repetition>\d{2})_(?P<action>[\w\-]+)_cam\d+_frames_(?P<start>\d+)_(?P<end>\d+)\.avi"
    )
    match = pattern.fullmatch(fname)
    if not match:
        raise ValueError(f"Filename format not recognized: {fname}")
    groups = match.groupdict()
    return {
        "subject_id": groups["subject_raw"],
        "subject_instance": int(groups["subject_num"]),
        "action": groups["action"],
        "start_frame": int(groups["start"]),
        "end_frame": int(groups["end"]),
        "repetition": int(groups["repetition"]),
        "video_id": fname
    }

def build_ixmas_json(video_dir):
    grouped = defaultdict(list)

    for fname in sorted(os.listdir(video_dir)):
        if not fname.endswith(".avi"):
            continue
        meta = parse_filename(fname)
        subject_name = meta["subject_id"]
        action = meta["action"]
        key = (subject_name, action)
        grouped[key].append(meta)

    subject_name_to_index = {}
    next_subject_index = 0
    json_clips = []

    for (subject_name, action), clips in grouped.items():
        if len(clips) != 6:
            raise ValueError(f"Expected 6 videos for ({subject_name}, {action}), got {len(clips)}")

        # Assign numeric subject index if not already done
        if subject_name not in subject_name_to_index:
            subject_name_to_index[subject_name] = next_subject_index
            next_subject_index += 1

        clips_sorted = sorted(clips, key=lambda x: (x["subject_instance"], x["repetition"]))

        for i, clip in enumerate(clips_sorted):
            split = (
                "train" if i < 3 else
                "val" if i == 3 else
                "test"
            )
            json_clips.append({
                "video_id": clip["video_id"],
                "clip_id": f"{clip['video_id'].replace('.avi','')}",
                "label": action,
                "subject": subject_name_to_index[subject_name],
                "scenario": "",  # IXMAS has no scenario information
                "start_frame": clip["start_frame"],
                "end_frame": clip["end_frame"],
                "split": split
            })

    return json_clips

if __name__ == "__main__":
    input_dir = "IXMAS"  # Path to folder containing .avi videos
    output_file = "ixmas_clips.json"

    clips = build_ixmas_json(input_dir)
    with open(output_file, "w") as f:
        json.dump(clips, f, indent=2)

    print(f"Saved {len(clips)} clips to {output_file}")
