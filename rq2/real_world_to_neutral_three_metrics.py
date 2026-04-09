import os
import json
import numpy as np
from scipy.spatial.distance import jensenshannon


def compute_race():
    # flux: {'KL': 0.1639, 'JS': 0.0603, 'TVD': 0.2399}
    # proteus: {'KL': 6.5474, 'JS': 0.5748, 'TVD': 0.8075}
    # sd3: {'KL': 0.1147, 'JS': 0.0411, 'TVD': 0.2264}
    # sana: {'KL': 3.5705, 'JS': 0.4298, 'TVD': 0.7155}
    # hunyuan: {'KL': 1.7918, 'JS': 0.3707, 'TVD': 0.6384}
    # qwen: {'KL': 0.213, 'JS': 0.0625, 'TVD': 0.2296}
    # kolors: {'KL': 1.9597, 'JS': 0.4712, 'TVD': 0.7365}
    # wan2.1: {'KL': 0.3596, 'JS': 0.1302, 'TVD': 0.3877}
    # ===== 路径改这里 =====
    race_json_dir = "/Users/wmt/projects/bias/write_into_json/race"

    # ===== 真实分布（已对齐为4类）=====
    P_real = np.array([
        0.3156,                     # Asian
        0.1628,                     # Black
        0.1515,                     # White
        0.1931 + 0.1770            # Others
    ])

    race_order = ["Asian", "Black", "White", "Others"]

    def normalize(arr):
        arr = np.array(arr, dtype=float)
        return arr / arr.sum()

    def compute_metrics(P_model, P_real):
        eps = 1e-8
        P_model = np.clip(P_model, eps, 1)
        P_real = np.clip(P_real, eps, 1)
        kl = np.sum(P_real * np.log(P_real / P_model))
        js = jensenshannon(P_real, P_model, base=2) ** 2
        tvd = 0.5 * np.abs(P_real - P_model).sum()

        return kl, js, tvd

    results = {}

    # ===== 遍历所有 json =====
    for file in os.listdir(race_json_dir):
        if not file.endswith("_race.json"):
            continue

        model_name = file.replace("_race.json", "")
        json_path = os.path.join(race_json_dir, file)
        print(model_name)
        with open(json_path, "r") as f:
            data = json.load(f)

        # ===== 只用 neutral =====
        neutral_counts = data["neutral"]

        P_model = np.array([
            neutral_counts.get("Asian", 0),
            neutral_counts.get("Black", 0),
            neutral_counts.get("White", 0),
            neutral_counts.get("Others", 0)
        ])

        P_model = normalize(P_model)

        kl, js, tvd = compute_metrics(P_model, P_real)

        results[model_name] = {
            "KL": round(kl, 4),
            "JS": round(js, 4),
            "TVD": round(tvd, 4)
        }

    # ===== 排序（按你论文顺序）=====
    order = ["flux", "proteus", "sd3", "sana", "hunyuan", "qwen", "kolors", "wan2.1"]

    for m in order:
        if m in results:
            print(f"{m}: {results[m]}")

def compute_age():
    # flux: {'KL': 0.8586, 'JS': 0.1744, 'TVD': 0.3958}
    # proteus: {'KL': 5.8002, 'JS': 0.3767, 'TVD': 0.6178}
    # sd3: {'KL': 5.6895, 'JS': 0.413, 'TVD': 0.6658}
    # sana: {'KL': 8.7227, 'JS': 0.3741, 'TVD': 0.5391}
    # hunyuan: {'KL': 5.1848, 'JS': 0.2947, 'TVD': 0.5298}
    # qwen: {'KL': 4.0543, 'JS': 0.3223, 'TVD': 0.5228}
    # kolors: {'KL': 1.0907, 'JS': 0.274, 'TVD': 0.5608}
    # wan2.1: {'KL': 4.0638, 'JS': 0.4102, 'TVD': 0.6658}
    # ===== 路径 =====
    age_json_dir = "/Users/wmt/projects/bias/write_into_json/age"

    # ===== 真实分布 =====
    P_real = np.array([
        0.165959,
        0.163733,
        0.297204,
        0.231504,
        0.141601
    ])

    age_order = ["0-9", "10-19", "20-39", "40-59", "60+"]

    def normalize(arr):
        arr = np.array(arr, dtype=float)
        return arr / arr.sum()

    def compute_metrics(P_model, P_real):
        eps = 1e-8
        P_model = np.clip(P_model, eps, 1)
        P_real = np.clip(P_real, eps, 1)

        # ⚠️ 注意你现在用的是 KL(P_real || P_model)
        kl = np.sum(P_real * np.log(P_real / P_model))

        js = jensenshannon(P_real, P_model, base=2) ** 2
        tvd = 0.5 * np.abs(P_real - P_model).sum()

        return kl, js, tvd

    results = {}

    # ===== 遍历 json =====
    for file in os.listdir(age_json_dir):
        if not file.endswith("_age.json"):
            continue

        model_name = file.replace("_age.json", "")
        json_path = os.path.join(age_json_dir, file)

        print(model_name)

        with open(json_path, "r") as f:
            data = json.load(f)

        if "neutral" not in data:
            print(f"Warning: {model_name} 没有 neutral")
            continue

        neutral_counts = data["neutral"]

        # ===== 按顺序取 =====
        P_model = np.array([
            neutral_counts.get(age, 0) for age in age_order
        ])

        P_model = normalize(P_model)

        kl, js, tvd = compute_metrics(P_model, P_real)

        results[model_name] = {
            "KL": round(kl, 4),
            "JS": round(js, 4),
            "TVD": round(tvd, 4)
        }

    # ===== 排序输出 =====
    order = ["flux", "proteus", "sd3", "sana", "hunyuan", "qwen", "kolors", "wan2.1"]

    print("\n=== AGE RESULTS ===")
    for m in order:
        if m in results:
            print(f"{m}: {results[m]}")

def compute_gender():
    # == = GENDER
    # RESULTS == =
    # flux: {'KL': 0.0392, 'JS': 0.0139, 'TVD': 0.1371}
    # proteus: {'KL': 1.0653, 'JS': 0.2349, 'TVD': 0.4671}
    # sd3: {'KL': 0.2768, 'JS': 0.0884, 'TVD': 0.3251}
    # sana: {'KL': 0.8486, 'JS': 0.2088, 'TVD': 0.4539}
    # hunyuan: {'KL': 0.2723, 'JS': 0.0874, 'TVD': 0.3249}
    # qwen: {'KL': 0.401, 'JS': 0.121, 'TVD': 0.3701}
    # kolors: {'KL': 0.0025, 'JS': 0.0009, 'TVD': 0.0351}
    # wan2.1: {'KL': 0.0013, 'JS': 0.0005, 'TVD': 0.0251}
    # ===== 路径 =====
    gender_json_dir = "/Users/wmt/projects/bias/write_into_json/gender"

    # ===== 真实分布 =====
    P_real = np.array([
        0.5029022778431522,   # male
        0.4970977221568478    # female
    ])

    gender_order = ["male", "female"]

    def normalize(arr):
        arr = np.array(arr, dtype=float)
        return arr / arr.sum()

    def compute_metrics(P_model, P_real):
        eps = 1e-8
        P_model = np.clip(P_model, eps, 1)
        P_real = np.clip(P_real, eps, 1)

        # KL(P_real || P_model)
        kl = np.sum(P_real * np.log(P_real / P_model))

        js = jensenshannon(P_real, P_model, base=2) ** 2
        tvd = 0.5 * np.abs(P_real - P_model).sum()

        return kl, js, tvd

    results = {}

    # ===== 遍历 json =====
    for file in os.listdir(gender_json_dir):
        if not file.endswith("_gender.json"):
            continue

        model_name = file.replace("_gender.json", "")
        json_path = os.path.join(gender_json_dir, file)

        print(model_name)

        with open(json_path, "r") as f:
            data = json.load(f)

        if "neutral" not in data:
            print(f"Warning: {model_name} 没有 neutral")
            continue

        neutral_counts = data["neutral"]

        # ⚠️ 注意大小写
        P_model = np.array([
            neutral_counts.get("Male", 0),
            neutral_counts.get("Female", 0)
        ])

        P_model = normalize(P_model)

        kl, js, tvd = compute_metrics(P_model, P_real)

        results[model_name] = {
            "KL": round(kl, 4),
            "JS": round(js, 4),
            "TVD": round(tvd, 4)
        }

    # ===== 排序输出 =====
    order = ["flux", "proteus", "sd3", "sana", "hunyuan", "qwen", "kolors", "wan2.1"]

    print("\n=== GENDER RESULTS ===")
    for m in order:
        if m in results:
            print(f"{m}: {results[m]}")

if __name__ == '__main__':
    compute_race()
    compute_age()
    compute_gender()
    print("done")