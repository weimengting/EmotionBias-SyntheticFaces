import os
import pandas as pd
import json

root_dir = "/Users/wmt/projects/bias/data/corrected"

age_groups = ["0-9", "10-19", "20-39", "40-59", "60+"]

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

        # ===== 初始化 =====
        counts = {age: 0 for age in age_groups}

        # ===== 统计 =====
        if "age_final" not in df.columns:
            print(f"Warning: {file} 没有 age_final")
            continue

        for val in df["age_final"]:
            if pd.isna(val):
                continue

            val = str(val).strip()

            if val in counts:
                counts[val] += 1

        result[emotion] = counts

    # ===== 保存 JSON =====
    save_root = "/Users/wmt/projects/bias/write_into_json/age"
    os.makedirs(save_root, exist_ok=True)
    out_path = os.path.join(save_root, f"{model_name}_age.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Done: {model_name}")

if __name__ == '__main__':
    print("done")