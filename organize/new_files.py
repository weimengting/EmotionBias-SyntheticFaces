import pandas as pd
import numpy as np
import ast
import os
import re


def extract_index(path):
    m = re.search(r'(\d{4})_face', str(path))
    if m:
        return int(m.group(1))
    return None


def merge_csv(original_csv, complete_csv, save_csv):

    df_orig = pd.read_csv(original_csv)
    df_comp = pd.read_csv(complete_csv)

    # 提取 index
    df_orig["index_id"] = df_orig["face_name_align"].apply(extract_index)
    df_comp["index_id"] = df_comp["face_name_align"].apply(extract_index)

    # 合并
    df_final = pd.concat([df_orig, df_comp], ignore_index=True)

    # 排序
    df_final = df_final.sort_values("index_id")

    # 删除辅助列
    df_final = df_final.drop(columns=["index_id"])

    df_final.to_csv(save_csv, index=False)

    print(f"Saved: {save_csv} ({len(df_final)} rows)")


def process_model(model_path):

    complete_path = os.path.join(model_path, "complete")

    if not os.path.exists(complete_path):
        return

    final_path = os.path.join(model_path, "final")
    os.makedirs(final_path, exist_ok=True)

    for file in os.listdir(complete_path):

        if not file.endswith(".csv"):
            continue

        if file.startswith("._"):
            continue

        complete_csv = os.path.join(complete_path, file)

        # persons -> neutral
        if file == "persons_res.csv":
            original_name = "neutral_res.csv"
        else:
            original_name = file

        original_csv = os.path.join(model_path, original_name)

        if not os.path.exists(original_csv):
            print("Missing original:", original_csv)
            continue

        save_csv = os.path.join(final_path, original_name)

        merge_csv(original_csv, complete_csv, save_csv)


def batch_process_insert(root_dir):

    for model in os.listdir(root_dir):

        model_path = os.path.join(root_dir, model)

        if not os.path.isdir(model_path):
            continue

        print("\nProcessing model:", model)

        process_model(model_path)

def count_rows(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return len(df)
    except:
        return 0


def check_complete(root_dir):

    for model in os.listdir(root_dir):

        model_path = os.path.join(root_dir, model)

        if not os.path.isdir(model_path):
            continue

        complete_path = os.path.join(model_path, "complete")

        if not os.path.exists(complete_path):
            continue

        print(f"\nChecking model: {model}")

        for file in os.listdir(complete_path):

            if not file.endswith(".csv"):
                continue

            if file.startswith("._"):
                continue

            complete_csv = os.path.join(complete_path, file)

            # 特殊规则
            if file == "persons_res.csv":
                original_name = "neutral_res.csv"
            else:
                original_name = file

            original_csv = os.path.join(model_path, original_name)

            if not os.path.exists(original_csv):
                print(f"Missing original file: {original_name}")
                continue

            original_count = count_rows(original_csv)
            complete_count = count_rows(complete_csv)

            total = original_count + complete_count

            if total != 1000:
                print(f"{file} ❌ total={total} (orig={original_count}, complete={complete_count})")
            else:
                print(f"{file} ✅ total=1000")



def race_remapping(csv_path):
    # csv_path = "/Users/wmt/projects/bias/data/res/wan2.1/unhappy_res.csv"

    df = pd.read_csv(csv_path)

    race_labels = [
        "White",
        "Black",
        "Latino_Hispanic",
        "East Asian",
        "Southeast Asian",
        "Indian",
        "Middle Eastern"
    ]

    def parse_scores(s):
        if isinstance(s, str):
            s = s.strip()[1:-1]  # 去掉 []
            return np.array([float(x) for x in s.split()])
        return np.array(s)

    def convert_race(row):
        race = row["race"]
        scores = parse_scores(row["race_scores_fair"])

        if race == "Middle Eastern":
            second_idx = np.argsort(scores)[-2]
            race = race_labels[second_idx]

        if race == "White":
            return "White"
        elif race == "Black":
            return "Black"
        elif race in ["East Asian", "Southeast Asian"]:
            return "Asian"
        elif race in ["Indian", "Latino_Hispanic"]:
            return "Others"
        else:
            return race

    df["race_final"] = df.apply(convert_race, axis=1)

    age_map = {
        "0-2": "0-9",
        "3-9": "0-9",
        "10-19": "10-19",
        "20-29": "20-39",
        "30-39": "20-39",
        "40-49": "40-59",
        "50-59": "40-59",
        "60-69": "60+",
        "70+": "60+"
    }

    # 新列
    df["age_final"] = df["age"].map(age_map)
    # 删除 face_name_align 不含 face0 的行
    df = df[df["face_name_align"].str.contains("face0", na=False)]
    # 覆盖写回原csv
    df.to_csv(csv_path, index=False)

    print("race_final column added.")

def batch_process(root_dir):

    for model in os.listdir(root_dir):

        model_path = os.path.join(root_dir, model)

        if not os.path.isdir(model_path):
            continue

        complete_path = os.path.join(model_path, "complete")

        if not os.path.exists(complete_path):
            continue

        for file in os.listdir(complete_path):
            if not file.endswith(".csv"):
                continue

            if file.startswith("._"):
                continue

            if file.endswith(".csv"):

                csv_path = os.path.join(complete_path, file)
                print(csv_path)
                race_remapping(csv_path)

def sort_all_files():
    import pandas as pd
    import os
    import re

    folder = "/Users/wmt/projects/bias/data/res/wan2.1"

    pattern = re.compile(r'_(\d{4})_face')
    csv_files = [
        "angry_res.csv",
        "disgusted_res.csv",
        "fearful_res.csv",
        "happy_res.csv",
        "neutral_res.csv",
        "sad_res.csv",
        "surprised_res.csv",
        "unhappy_res.csv"
    ]
    for file in csv_files:
        csv_path = os.path.join(folder, file)
        df = pd.read_csv(csv_path)

        # 提取编号
        df["index_id"] = df["face_name_align"].str.extract(r'_(\d{4})_face')

        # 2 打印没有匹配到编号的行
        print(df[df["index_id"].isna()][["face_name_align"]])
        df["index_id"] = df["index_id"].astype(int)
        # 排序
        df = df.sort_values("index_id").reset_index(drop=True)

        # 找缺失编号
        expected = set(range(1000))
        existing = set(df["index_id"].tolist())
        missing = sorted(expected - existing)

        # 删除辅助列
        df = df.drop(columns=["index_id"])

        # 覆盖写回csv
        df.to_csv(csv_path, index=False)

        # 写missing txt
        txt_path = csv_path.replace(".csv", "_missing.txt")
        with open(txt_path, "w") as f:
            for m in missing:
                f.write(f"{m:04d}\n")

        print(file, "missing:", len(missing))



if __name__ == '__main__':
    # race_remapping()
    # batch_process("/Volumes/KINGSTON/projects/bias")
    root_dir = "/Volumes/KINGSTON/projects/bias"
    # check_complete(root_dir)
    batch_process_insert(root_dir)
    # sort_all_files()
    # print("done")