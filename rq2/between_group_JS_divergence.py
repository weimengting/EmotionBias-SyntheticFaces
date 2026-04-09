import os
import json
import numpy as np
from itertools import combinations
from scipy.spatial.distance import jensenshannon
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_age():
    base_dir = "/Users/wmt/projects/bias/write_into_json/age"  # 改成你的路径
    # ===== 2. 模型分组（你自己确认）=====
    western_models = ["flux", "sd3", "sana", "proteus"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]


    # ===== 3. 读取 JSON（只取 neutral）=====
    def load_neutral_distribution(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)

        neutral_data = data["neutral"]

        keys = ["0-9", "10-19", "20-39", "40-59", "60+"]
        vec = np.array([neutral_data[k] for k in keys], dtype=float)
        vec = vec / vec.sum()

        return vec


    model_distributions = {}

    for filename in os.listdir(base_dir):
        if filename.endswith(".json"):
            model_name = filename.replace("_age.json", "")
            path = os.path.join(base_dir, filename)
            model_distributions[model_name] = load_neutral_distribution(path)


    # ===== 5. JS divergence =====
    def js_divergence(p, q):
        return jensenshannon(p, q, base=2) ** 2


    # ===== 6. 分组 =====
    western_vecs = [model_distributions[m] for m in western_models]
    chinese_vecs = [model_distributions[m] for m in chinese_models]

    # ===== 7. aggregate 分布 =====
    P_w = np.mean(western_vecs, axis=0)
    P_c = np.mean(chinese_vecs, axis=0)

    # ===== 8. between-group JS =====
    js_between = js_divergence(P_w, P_c)


    # ===== 9. within-group JS =====
    def mean_pairwise_js(group):
        js_list = []
        for p, q in combinations(group, 2):
            js_list.append(js_divergence(p, q))
        return np.mean(js_list)


    W1 = mean_pairwise_js(western_vecs)
    W2 = mean_pairwise_js(chinese_vecs)

    # ===== 10. 输出 =====
    print("=== Age (Neutral Only) ===")
    print(f"Between-group JS: {js_between:.4f}")
    print(f"Within Western JS: {W1:.4f}")
    print(f"Within Chinese JS: {W2:.4f}")

def compute_gender():
    base_dir = "/Users/wmt/projects/bias/write_into_json/gender"  # 改成你的路径

    # ===== 2. 模型分组 =====
    western_models = ["flux", "sd3", "sana", "proteus"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]

    # ===== 3. 读取 JSON（只取 neutral）=====
    def load_neutral_gender(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)

        neutral_data = data["neutral"]  # 只取 neutral

        # 注意顺序固定！！
        keys = ["Male", "Female"]
        vec = np.array([neutral_data[k] for k in keys], dtype=float)

        # 归一化
        vec = vec / vec.sum()

        return vec

    # ===== 4. 加载所有模型 =====
    model_distributions = {}

    for filename in os.listdir(base_dir):
        if filename.endswith(".json"):
            model_name = filename.replace("_gender.json", "")
            path = os.path.join(base_dir, filename)
            model_distributions[model_name] = load_neutral_gender(path)

    # ===== 5. JS divergence =====
    def js_divergence(p, q):
        return jensenshannon(p, q, base=2) ** 2

    # ===== 6. 分组 =====
    western_vecs = [model_distributions[m] for m in western_models]
    chinese_vecs = [model_distributions[m] for m in chinese_models]

    # ===== 7. aggregate =====
    P_w = np.mean(western_vecs, axis=0)
    P_c = np.mean(chinese_vecs, axis=0)

    # ===== 8. between-group =====
    js_between = js_divergence(P_w, P_c)

    # ===== 9. within-group =====
    def mean_pairwise_js(group):
        js_list = []
        for p, q in combinations(group, 2):
            js_list.append(js_divergence(p, q))
        return np.mean(js_list)

    W1 = mean_pairwise_js(western_vecs)
    W2 = mean_pairwise_js(chinese_vecs)

    # ===== 10. 输出 =====
    print("=== Gender (Neutral Only) ===")
    print(f"Between-group JS: {js_between:.4f}")
    print(f"Within Western JS: {W1:.4f}")
    print(f"Within Chinese JS: {W2:.4f}")


def compute_race():
    base_dir = "/Users/wmt/projects/bias/write_into_json/race"  # 改成你的路径

    # ===== 2. 模型分组 =====
    western_models = ["flux", "sd3", "sana", "proteus"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]

    # ===== 3. 读取 JSON（只取 neutral）=====
    def load_neutral_race(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)

        neutral_data = data["neutral"]  # 只取 neutral

        # ⚠️ 顺序必须固定！
        keys = ["White", "Black", "Asian", "Others"]
        vec = np.array([neutral_data[k] for k in keys], dtype=float)

        # 归一化
        vec = vec / vec.sum()

        return vec

    # ===== 4. 加载所有模型 =====
    model_distributions = {}

    for filename in os.listdir(base_dir):
        if filename.endswith(".json"):
            model_name = filename.replace("_race.json", "")
            path = os.path.join(base_dir, filename)
            model_distributions[model_name] = load_neutral_race(path)

    # ===== 5. JS divergence =====
    def js_divergence(p, q):
        return jensenshannon(p, q, base=2) ** 2

    # ===== 6. 分组 =====
    western_vecs = [model_distributions[m] for m in western_models]
    chinese_vecs = [model_distributions[m] for m in chinese_models]

    # ===== 7. aggregate =====
    P_w = np.mean(western_vecs, axis=0)
    P_c = np.mean(chinese_vecs, axis=0)

    # ===== 8. between-group =====
    js_between = js_divergence(P_w, P_c)

    # ===== 9. within-group =====
    def mean_pairwise_js(group):
        js_list = []
        for p, q in combinations(group, 2):
            js_list.append(js_divergence(p, q))
        return np.mean(js_list)

    W1 = mean_pairwise_js(western_vecs)
    W2 = mean_pairwise_js(chinese_vecs)

    # ===== 10. 输出 =====
    print("=== Race (Neutral Only) ===")
    print(f"Between-group JS: {js_between:.4f}")
    print(f"Within Western JS: {W1:.4f}")
    print(f"Within Chinese JS: {W2:.4f}")

def plot_res():
    # ===== 数据整理（long format）=====
    data = {
        "Attribute": ["Age", "Age", "Age",
                      "Gender", "Gender", "Gender",
                      "Race", "Race", "Race"],
        "Type": ["Between-group", "Within Western", "Within Chinese"] * 3,
        "JS": [0.0309, 0.2194, 0.0487,
               0.0063, 0.3025, 0.1334,
               0.0229, 0.3059, 0.4640]
    }

    df = pd.DataFrame(data)

    # ===== 风格 =====
    sns.set(style="whitegrid")

    # ===== 画图 =====
    plt.figure(figsize=(6, 4))

    sns.barplot(
        data=df,
        x="Attribute",
        y="JS",
        hue="Type"
    )

    # ===== 标签 =====
    plt.ylabel("JS Divergence")
    plt.xlabel("")

    # ===== 图例 =====
    plt.legend(title="")

    # ===== 紧凑 =====
    plt.tight_layout()

    # ===== 保存 =====
    plt.savefig("js_seaborn.png", dpi=300)
    plt.show()

# === Age (Neutral Only) ===
# Between-group JS: 0.0309
# Within Western JS: 0.2194
# Within Chinese JS: 0.0487

# === Gender (Neutral Only) ===
# Between-group JS: 0.0063
# Within Western JS: 0.3025
# Within Chinese JS: 0.1334

# === Race (Neutral Only) ===
# Between-group JS: 0.0229
# Within Western JS: 0.3059
# Within Chinese JS: 0.4640


if __name__ == '__main__':
    # compute_age()
    # compute_gender()
    # compute_race()
    plot_res()