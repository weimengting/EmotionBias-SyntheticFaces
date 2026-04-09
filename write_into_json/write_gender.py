import os
import pandas as pd
import json

root_dir = "/Users/wmt/projects/bias/data/corrected"  # 你的总目录

# ===== 遍历每个模型 =====
for model_name in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model_name)

    if not os.path.isdir(model_path):
        continue

    result = {}

    # ===== 遍历每个 CSV =====
    for file in os.listdir(model_path):
        if not file.endswith(".csv"):
            continue

        emotion = file.replace("_res.csv", "")  # angry, happy...

        csv_path = os.path.join(model_path, file)
        df = pd.read_csv(csv_path)

        # ===== 自动找 gender 列（更稳）=====
        # 假设列名包含 "Male"/"Female" 或类似
        male_count = 0
        female_count = 0

        for col in df.columns:
            if "male" in col.lower():
                male_count = df[col].sum()
            elif "female" in col.lower():
                female_count = df[col].sum()

        # ===== 如果是分类标签形式（更常见）=====
        if male_count == 0 and female_count == 0:
            if "gender" in df.columns:
                male_count = (df["gender"].str.lower() == "male").sum()
                female_count = (df["gender"].str.lower() == "female").sum()

        result[emotion] = {
            "Male": int(male_count),
            "Female": int(female_count)
        }
    save_root = "/Users/wmt/projects/bias/write_into_json/gender"
    os.makedirs(save_root, exist_ok=True)
    # ===== 保存 JSON =====
    out_path = os.path.join(save_root, f"{model_name}_gender.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Done: {model_name}")

if __name__ == '__main__':
    print("done")