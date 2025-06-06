import json

# Define 6 IXMAS action classes to keep 
allowed_classes = {
    "check_watch", "point", "kick",
    "sit_down", "get_up", "walk"
}

# Load the list-based JSON
with open("ixmas_clips.json", "r") as f:
    data = json.load(f)

# Filter the list
filtered_data = [item for item in data if item.get("label") in allowed_classes]

# Save the filtered result
with open("ixmas_clips_6.json", "w") as f:
    json.dump(filtered_data, f, indent=2)

print(f"Saved {len(filtered_data)} clips with selected classes.")
