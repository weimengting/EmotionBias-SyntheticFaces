import os
import pandas as pd
import json
import numpy as np
import ast

root_dir = "/Users/wmt/projects/bias/data/corrected"

race_order = [
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern"
]

def get_second_largest_index(arr):
    arr = np.array(arr)
    return arr.argsort()[-2]

def group_race(label):
    if label in ["East Asian", "Southeast Asian"]:
        return "Asian"
    elif label in ["Indian", "Latino_Hispanic", "Middle Eastern"]:
        return "Others"
    else:
        return label  # White / Black

for model_name in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model_name)

    if not os.path.isdir(model_path):
        continue

    result = {}

    for file in os.listdir(model_path):
        if not file.endswith(".csv"):
            continue

        emotion = file.replace("_res.csv", "")
        csv_path = os.path.join(model_path, file)

        df = pd.read_csv(csv_path)

        counts = {
            "White": 0,
            "Black": 0,
            "Asian": 0,
            "Others": 0
        }

        for _, row in df.iterrows():

            original_race = row["race"]

            # ===== 只对 Middle Eastern 重新映射 =====
            if str(original_race).lower() == "middle eastern":

                scores = row["race_scores_fair"]
                if isinstance(scores, str):
                    scores = np.fromstring(scores.strip("[]"), sep=" ")

                idx = get_second_largest_index(scores)
                final_label = race_order[idx]

            else:
                final_label = original_race

            # ===== grouping =====
            final_group = group_race(final_label)

            counts[final_group] += 1

        result[emotion] = counts
    save_root = "/Users/wmt/projects/bias/write_into_json/race"
    os.makedirs(save_root, exist_ok=True)
    out_path = os.path.join(save_root, f"{model_name}_race.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Done: {model_name}")

if __name__ == '__main__':
    print("done")