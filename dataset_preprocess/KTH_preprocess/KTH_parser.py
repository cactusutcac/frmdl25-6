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
                    parsed.append({
                        "video_id": video_id,
                        "clip_id": f"{video_id}_clip{idx+1}",
                        "label": label,
                        "subject": subject,
                        "scenario": scenario,
                        "start_frame": start,
                        "end_frame": end
                    })
    return parsed

if __name__ == "__main__":
    input_path = "KTH/sequences.txt"
    output_path = "kth_clips.json"

    result = parse_sequences_txt(input_path)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
