import os
import json
import pandas as pd


# unhappy的里面很少有这个
def build_attractiveness_jsons():
    # 你的总目录
    root_dir = "/Users/wmt/projects/bias/data/corrected"

    # 8 个模型目录
    model_dirs = [
        "flux",
        "hunyuan",
        "kolors",
        "proteus",
        "qwen",
        "sana",
        "sd3",
        "wan2.1",
    ]

    # 情绪文件名
    emotion_files = [
        "angry_res.csv",
        "disgusted_res.csv",
        "fearful_res.csv",
        "happy_res.csv",
        "neutral_res.csv",
        "sad_res.csv",
        "surprised_res.csv",
        "unhappy_res.csv",
    ]

    for model in model_dirs:
        model_path = os.path.join(root_dir, model)
        result_dict = {}

        for csv_name in emotion_files:
            csv_path = os.path.join(model_path, csv_name)

            if not os.path.exists(csv_path):
                print(f"[Warning] File not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            if "attractiveness" not in df.columns:
                print(f"[Warning] 'attractiveness' column not found in: {csv_path}")
                continue

            emotion = csv_name.replace("_res.csv", "")

            # 统计 attractiveness 每个值的数量
            counts = (
                df["attractiveness"]
                .value_counts(dropna=False)
                .sort_index()
            )

            # 转成普通 dict，并把 key 转成字符串，方便写 json
            result_dict[emotion] = {
                str(k): int(v) for k, v in counts.items()
            }

        save_root = "/Users/wmt/projects/bias/write_into_json/attractiveness"
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, f"{model}_attractiveness.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    build_attractiveness_jsons()