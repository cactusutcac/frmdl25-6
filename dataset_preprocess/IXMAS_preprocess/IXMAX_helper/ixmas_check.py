import json
import pandas as pd 
from collections import Counter

# Load the IXMAS clip metadata from the uploaded JSON file
with open("ixmas_clips.json", "r") as f:
    ixmas_data = json.load(f)

# Extract and sort all unique action labels
unique_labels = sorted(set(item["label"] for item in ixmas_data))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

# Count distribution of each label
label_counts = Counter(item["label"] for item in ixmas_data)

# Prepare data for review
label_info = {
    "Label": unique_labels,
    "Mapped Index": [label_to_index[label] for label in unique_labels],
    "Count": [label_counts[label] for label in unique_labels]
}

df_labels = pd.DataFrame(label_info)
print(df_labels)

#            Label  Mapped Index  Count
# 0    check-watch             0     60
# 1     cross-arms             1     60
# 2         get-up             2     60
# 3           kick             3     60
# 4        pick-up             4     60
# 5          point             5     60
# 6          punch             6     60
# 7   scratch-head             7     60
# 8       sit-down             8     60
# 9    turn-around             9     60
# 10          walk            10     60
# 11          wave            11     60