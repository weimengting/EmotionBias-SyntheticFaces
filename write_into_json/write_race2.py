import os
import pandas as pd
import json

root_dir = "/Users/wmt/projects/bias/data/corrected"
save_root = "/Users/wmt/projects/bias/write_into_json/race2"

os.makedirs(save_root, exist_ok=True)

valid_races = ["White", "Black", "Asian", "Others"]

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

        counts = {r: 0 for r in valid_races}

        if "race_final" not in df.columns:
            print(f"Warning: {file} 没有 race_final")
            continue

        for val in df["race_final"]:
            if pd.isna(val):
                continue

            val = str(val).strip()

            if val in counts:
                counts[val] += 1
            else:
                print("Unknown race:", val)  # debug用

        result[emotion] = counts

    out_path = os.path.join(save_root, f"{model_name}_race.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Done: {model_name}")

if __name__ == '__main__':

    print("All done!")
