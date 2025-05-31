import re
import json

def parse_key(key):
    """Extract subject, label, and scenario from the video name."""
    match = re.match(r"person(\d{2})_([a-z]+)_d(\d)", key)
    if match:
        subject, label, scenario = match.groups()
        return label, subject, scenario
    else:
        raise ValueError(f"Unexpected format in key: {key}")

def parse_sequences_txt(txt_path):
    """Parse sequences.txt and return a list of annotated clip dictionaries."""
    with open(txt_path, "r") as file:
        lines = file.readlines()

    pattern = re.compile(r"(\w+)\s+frames\s+([\d\-, ]+)")
    parsed = []

    for line in lines:
        match = pattern.search(line)
        if match:
            video_id = match.group(1)
            frame_ranges = match.group(2).split(',')
            label, subject, scenario = parse_key(video_id)

            for idx, fr in enumerate(frame_ranges):
                fr = fr.strip()
                if '-' in fr:
                    start, end = map(int, fr.split('-'))
                    if abs(start - end) < 32:
                        continue  # Skip sequences shorter than 32 frames (messes up whole experiment)
                    parsed.append({
                        "video_id": video_id,
                        "clip_id": f"{video_id}_clip{idx+1}",
                        "label": label,
                        "subject": int(subject)-1,
                        "scenario": scenario,
                        "start_frame": start,
                        "end_frame": end,
                        "split": "train" if idx == 0 or idx == 1 else "val" if idx == 2 else "test" if idx == 3 else "" # 1st and 2nd clips used for training, 3rd for validation, 4th for testing
                    })
    return parsed

if __name__ == "__main__":
    input_path = "KTH/00sequences.txt"
    output_path = "kth_clips.json"

    result = parse_sequences_txt(input_path)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
