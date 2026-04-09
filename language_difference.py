import pandas as pd
import numpy as np
import ast
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_race():
    # ============================
    # 1. 读取数据
    # ============================
    csv_path = "/Users/wmt/projects/bias/data/res/qwen/chinese_person_res.csv"
    df = pd.read_csv(csv_path)

    # ============================
    # 2. 筛选 face0 行
    # ============================
    mask_face0 = df["face_name_align"].astype(str).str.contains("face0", na=False)
    df_face0 = df[mask_face0].copy()

    # ============================
    # 3. race_scores_fair 顺序
    # ============================
    race_labels = [
        "White",
        "Black",
        "Latino_Hispanic",
        "East Asian",
        "Southeast Asian",
        "Indian",
        "Middle Eastern"
    ]

    def parse_scores(scores):
        """
        解析 race_scores_fair：
        支持 '[0.1 0.2 0.3]' 和 '[0.1, 0.2, 0.3]' 两种格式
        """
        if isinstance(scores, (list, tuple, np.ndarray)):
            return list(scores)

        if isinstance(scores, str):
            s = scores.strip()

            # 如果是 numpy 风格: "[0.1 0.2 0.3]"
            if "[" in s and "]" in s and "," not in s:
                s_clean = s.replace("[", "").replace("]", "")
                return [float(x) for x in s_clean.split()]

            # 否则尝试 literal_eval
            try:
                return list(ast.literal_eval(s))
            except Exception:
                raise ValueError(f"无法解析 race_scores_fair: {scores}")

        raise ValueError(f"未知格式: {scores}")

    # ============================
    # 4. Middle Eastern 变成第二高 + 合并逻辑
    # ============================
    def get_merged_race(row):
        race = row["race"]

        # ------- Step 1: Middle Eastern → 第二高类别 -------
        if race == "Middle Eastern":
            scores = parse_scores(row["race_scores_fair"])
            scores_arr = np.array(scores, dtype=float)

            # 找第二高（不会是 Middle Eastern）
            sorted_idx = np.argsort(scores_arr)
            print(sorted_idx)
            idx_second = sorted_idx[-2]
            race = race_labels[idx_second]
            print('new race', race)

        # ------- Step 2: 合并 -------
        if race in ["Latino_Hispanic", "Indian"]:
            return "Others"
        if race in ["East Asian", "Southeast Asian"]:
            return "Asian"
        if race in ["White", "Black"]:
            return race

        # 保险起见（理论上不会走到这里）
        return race

    # ============================
    # 5. 添加合并后的种族
    # ============================
    df_face0["race_merged"] = df_face0.apply(get_merged_race, axis=1)
    df.loc[mask_face0, "race_merged"] = df_face0["race_merged"]

    # ============================
    # 6. 统计并存 JSON
    # ============================
    race_counts = df_face0["race_merged"].value_counts().to_dict()

    output_json = {
        "persons": race_counts
    }

    json_path = "/Users/wmt/projects/bias/data/jsons/race/qwen_persons_race_count_CN.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=4)

    print("完成！不可能出现 Middle Eastern。")
    print(json.dumps(output_json, indent=4))


def preprocess_gender():
    # ============================
    # 1. 读取数据
    # ============================
    csv_path = "/Users/wmt/projects/bias/data/res/wan2.1/chinese_person_res.csv"
    df = pd.read_csv(csv_path)

    # ============================
    # 2. 筛选 face0 行（保持一致）
    # ============================
    mask_face0 = df["face_name_align"].astype(str).str.contains("face0", na=False)
    df_face0 = df[mask_face0].copy()

    # ============================
    # 3. 统一 gender 标签格式
    # ============================
    # 如果列名不是 gender，请告诉我
    df_face0["gender"] = df_face0["gender"].astype(str).str.capitalize()

    # 可选：如果你的 gender 是 M/F，可以映射为 Male / Female
    mapping = {
        "m": "Male",
        "f": "Female",
        "male": "Male",
        "female": "Female",
        "0": "Female",
        "1": "Male"
    }
    df_face0["gender"] = df_face0["gender"].map(lambda x: mapping.get(x.lower(), x))

    # ============================
    # 4. 统计性别数量
    # ============================
    gender_counts = df_face0["gender"].value_counts().to_dict()

    # ============================
    # 5. 生成 JSON
    # ============================
    output_json = {
        "persons": gender_counts
    }

    json_path = "/Users/wmt/projects/bias/data/jsons/gender/wan2.1_persons_gender_count_CN.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=4)

    print("性别统计完成！")
    print(json.dumps(output_json, indent=4))


def preprocess_age():
    # ============================
    # 1. 读取数据
    # ============================
    csv_path = "/Users/wmt/projects/bias/data/res/hunyuan/chinese_person_res.csv"
    df = pd.read_csv(csv_path)

    # ============================
    # 2. 筛选 face0 行
    # ============================
    mask_face0 = df["face_name_align"].astype(str).str.contains("face0", na=False)
    df_face0 = df[mask_face0].copy()

    # ============================
    # 3. 解析年龄区间，例如 "20-29" → 20
    # ============================
    age_col = "age"  # 你 CSV 的年龄列名

    def extract_lower_bound(age_str):
        """
        输入: "20-29" / "60+" / "0-9"
        输出: 20 / 60 / 0
        """
        if pd.isna(age_str):
            return None

        text = str(age_str)
        nums = re.findall(r"\d+", text)

        if len(nums) == 0:
            return None

        return int(nums[0])  # 取区间左端点作为年龄判定依据

    df_face0["age_lower"] = df_face0[age_col].apply(extract_lower_bound)

    df_face0 = df_face0.dropna(subset=["age_lower"])

    # ============================
    # 4. 映射到 5 大年龄段
    # ============================
    def map_age_group(lower):
        if lower <= 9:
            return "0-9"
        elif lower <= 19:
            return "10-19"
        elif lower <= 39:
            return "20-39"
        elif lower <= 59:
            return "40-59"
        else:
            return "60+"

    df_face0["age_group"] = df_face0["age_lower"].apply(map_age_group)

    # ============================
    # 5. 统计数量（保持顺序）
    # ============================
    age_bins = ["0-9", "10-19", "20-39", "40-59", "60+"]
    counts_raw = df_face0["age_group"].value_counts().to_dict()
    age_counts = {grp: counts_raw.get(grp, 0) for grp in age_bins}

    # ============================
    # 6. 输出 JSON
    # ============================
    output_json = {"persons": age_counts}

    output_path = "/Users/wmt/projects/bias/data/jsons/age/hunyuan_persons_age_count_CN.json"
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=4)

    print("年龄段统计完成！")
    print(json.dumps(output_json, indent=4))

def en_cn_race():
    import json, os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # ============================
    # 1. 配置
    # ============================
    races = ["White", "Black", "Asian", "Others"]
    models = ["hunyuan", "kolors", "qwen", "wan2.1"]
    languages = ["English", "Chinese"]

    paths = {
        "hunyuan": {
            "English": "/Users/wmt/projects/bias/data/jsons/race/hunyuan_emotion_race_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/race/hunyuan_persons_race_count_CN.json"
        },
        "kolors": {
            "English": "/Users/wmt/projects/bias/data/jsons/race/kolors_emotion_race_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/race/kolors_persons_race_count_CN.json"
        },
        "qwen": {
            "English": "/Users/wmt/projects/bias/data/jsons/race/qwen_emotion_race_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/race/qwen_persons_race_count_CN.json"
        },
        "wan2.1": {
            "English": "/Users/wmt/projects/bias/data/jsons/race/Wan2.1_emotion_race_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/race/wan2.1_persons_race_count_CN.json"
        }
    }

    # ============================
    # 2. 颜色设定（保留原逻辑）
    # ============================
    english_colors = ["#08306B", "#2171B5", "#6BAED6", "#C6DBEF"]

    chinese_colors = {
        "hunyuan": "#7F2704",
        "kolors": "#A63603",
        "qwen": "#E6550D",
        "wan2.1": "#FDBE85",
    }

    display_names = {
        "hunyuan": "Hunyuan",
        "kolors": "Kolors",
        "qwen": "Qwen",
        "wan2.1": "Wan2.1",
    }

    # ============================
    # 3. 读取数据
    # ============================
    proportions = {race: {model: {} for model in models} for race in races}

    for model in models:
        for lang in languages:
            with open(paths[model][lang], "r") as f:
                data = json.load(f)
            counts = data["person"]
            total = sum(counts.values())
            for race in races:
                proportions[race][model][lang] = counts.get(race, 0) / total

    # ============================
    # 4. x 坐标
    # ============================
    num_races = len(races)
    num_models = len(models)

    bar_width = 0.06
    model_group_width = 2 * bar_width + 0.04
    race_gap = 0.4

    race_centers = np.arange(num_races) * (num_models * model_group_width + race_gap)

    fig, ax = plt.subplots(figsize=(10, 6))

    # ============================
    # 5. 画柱子
    # ============================
    for r_idx, race in enumerate(races):
        center = race_centers[r_idx]
        total_width = num_models * model_group_width
        start = center - total_width / 2.0

        for m_idx, model in enumerate(models):
            group_center = start + m_idx * model_group_width + model_group_width / 2.0

            x_en = group_center - bar_width / 2.0
            x_cn = group_center + bar_width / 2.0

            en_val = proportions[race][model]["English"]
            cn_val = proportions[race][model]["Chinese"]

            ax.bar(x_en, en_val, width=bar_width,
                   color=english_colors[m_idx],
                   label=f"{display_names[model]} EN")
            ax.bar(x_cn, cn_val, width=bar_width,
                   color=chinese_colors[model],
                   label=f"{display_names[model]} CN")

    # ============================
    # 6. 轴 & legend（字体统一放大）
    # ============================
    ax.set_xticks(race_centers)
    ax.set_xticklabels(races, fontsize=27)
    ax.set_ylabel("Proportion", fontsize=23)

    ax.tick_params(axis="both", labelsize=27)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=18,          # 👈 legend 字体
        handlelength=1.4,     # 👈 图标宽度
        handleheight=0.9,     # 👈 图标高度
        markerscale=0.7,      # 👈 图标缩放
        bbox_to_anchor=(0.6, 1),
        loc="upper left",
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("race_distribution_models_EN_CN.png", dpi=300, bbox_inches="tight")
    plt.show()

def en_cn_race2():
    import json, os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns

    sns.set_style("whitegrid")

    # ============================
    # 1. 配置
    # ============================
    races = ["White", "Black", "Asian", "Others"]
    models = ["hunyuan", "kolors", "qwen", "wan2.1"]
    languages = ["English", "Chinese"]

    paths = {
        "hunyuan": {
            "English": "/Users/wmt/projects/bias/data/jsons/race/hunyuan_emotion_race_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/race/hunyuan_persons_race_count_CN.json"
        },
        "kolors": {
            "English": "/Users/wmt/projects/bias/data/jsons/race/kolors_emotion_race_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/race/kolors_persons_race_count_CN.json"
        },
        "qwen": {
            "English": "/Users/wmt/projects/bias/data/jsons/race/qwen_emotion_race_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/race/qwen_persons_race_count_CN.json"
        },
        "wan2.1": {
            "English": "/Users/wmt/projects/bias/data/jsons/race/Wan2.1_emotion_race_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/race/wan2.1_persons_race_count_CN.json"
        }
    }

    # ============================
    # 2. 配色：每个模型一个颜色，EN/CN 靠背景区分
    # ============================
    model_colors = {
        "hunyuan": "#7F2704",
        "kolors": "#D94801",
        "qwen": "#FD8D3C",
        "wan2.1": "#FDD0A2",
    }

    display_names = {
        "hunyuan": "Hunyuan",
        "kolors": "Kolors",
        "qwen": "Qwen",
        "wan2.1": "Wan2.1",
    }

    # ============================
    # 3. 读取数据
    # ============================
    proportions = {race: {model: {} for model in models} for race in races}

    for model in models:
        for lang in languages:
            with open(paths[model][lang], "r") as f:
                data = json.load(f)
            counts = data["person"]
            total = sum(counts.values())
            for race in races:
                proportions[race][model][lang] = counts.get(race, 0) / total

    # ============================
    # 4. x 坐标
    # ============================
    num_races = len(races)
    num_models = len(models)

    bar_width = 0.06
    model_group_width = 2 * bar_width + 0.02  # EN + CN + 小间距
    intra_gap = 0.12  # EN/CN 两组之间的间距
    race_gap = 0.5  # race 组之间的间距

    # 每个 race 组总宽 = num_models 个 model_group + (num_models-1) 个 intra_gap
    race_group_width = num_models * model_group_width + (num_models - 1) * intra_gap
    race_centers = np.arange(num_races) * (race_group_width + race_gap)

    fig, ax = plt.subplots(figsize=(12, 6))

    # ============================
    # 5. 背景色区分 EN / CN
    # ============================
    for r_idx in range(num_races):
        center = race_centers[r_idx]
        group_start = center - race_group_width / 2.0

        for m_idx in range(num_models):
            offset = m_idx * (model_group_width + intra_gap)
            group_center = group_start + offset + model_group_width / 2.0

            en_x = group_center - bar_width / 2.0
            cn_x = group_center + bar_width / 2.0

            # EN 浅蓝背景，CN 浅橙背景（更明显）
            ax.axvspan(en_x - bar_width * 0.6, en_x + bar_width * 0.6,
                       alpha=0.15, color="#4C8BE8", zorder=0)
            ax.axvspan(cn_x - bar_width * 0.6, cn_x + bar_width * 0.6,
                       alpha=0.15, color="#D94801", zorder=0)

    # ============================
    # 6. 画柱子
    # ============================
    for r_idx, race in enumerate(races):
        center = race_centers[r_idx]
        group_start = center - race_group_width / 2.0

        for m_idx, model in enumerate(models):
            offset = m_idx * (model_group_width + intra_gap)
            group_center = group_start + offset + model_group_width / 2.0

            en_x = group_center - bar_width / 2.0
            cn_x = group_center + bar_width / 2.0

            en_val = proportions[race][model]["English"]
            cn_val = proportions[race][model]["Chinese"]

            ax.bar(en_x, en_val, width=bar_width,
                   color=model_colors[model], alpha=0.88,
                   edgecolor="white", linewidth=0.4, zorder=3)
            ax.bar(cn_x, cn_val, width=bar_width,
                   color=model_colors[model], alpha=0.88,
                   edgecolor="white", linewidth=0.4, zorder=3)

    # ============================
    # 7. 轴
    # ============================
    ax.set_xticks(race_centers)
    ax.set_xticklabels(races, fontsize=27)
    ax.set_ylabel("Proportion", fontsize=23)
    ax.tick_params(axis="both", labelsize=27)
    ax.set_xlim(
        race_centers[0] - race_group_width / 2.0 - bar_width * 2,
        race_centers[-1] + race_group_width / 2.0 + bar_width * 2
    )

    # ============================
    # 8. 图例：4 模型颜色 + EN/CN 背景色说明
    # ============================
    en_patch = mpatches.Patch(color="#4C8BE8", alpha=0.4, label="English")
    cn_patch = mpatches.Patch(color="#D94801", alpha=0.4, label="Chinese")

    model_handles = [
        mpatches.Patch(color=model_colors[m], label=display_names[m])
        for m in models
    ]

    # 第一列：语言背景；第二列：模型颜色
    final_handles = [en_patch, model_handles[0], model_handles[1],
                     cn_patch, model_handles[2], model_handles[3]]
    final_labels = ["EN", display_names[models[0]], display_names[models[1]],
                    "CN", display_names[models[2]], display_names[models[3]]]

    leg = ax.legend(
        final_handles, final_labels,
        ncol=2,
        fontsize=15,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=0.9,
        handlelength=1.2,
        handleheight=1.0,
        borderpad=0.7,
        labelspacing=0.4,
        columnspacing=1.0,
    )

    for text in leg.get_texts():
        if text.get_text() in ("EN", "CN"):
            text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig("race_distribution_models_EN_CN.png", dpi=300, bbox_inches="tight")
    plt.show()

def en_cn_gender2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns

    sns.set_style("whitegrid")

    # ============================
    # 1. 配置
    # ============================
    genders   = ["Male", "Female"]
    models    = ["hunyuan", "kolors", "qwen", "wan2.1"]
    languages = ["English", "Chinese"]

    paths = {
        "hunyuan": {
            "English": "/Users/wmt/projects/bias/data/jsons/gender/hunyuan_gender_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/gender/hunyuan_persons_gender_count_CN.json"
        },
        "kolors": {
            "English": "/Users/wmt/projects/bias/data/jsons/gender/kolors_gender_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/gender/kolors_persons_gender_count_CN.json"
        },
        "qwen": {
            "English": "/Users/wmt/projects/bias/data/jsons/gender/qwen_gender_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/gender/qwen_persons_gender_count_CN.json"
        },
        "wan2.1": {
            "English": "/Users/wmt/projects/bias/data/jsons/gender/Wan2.1_gender_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/gender/wan2.1_persons_gender_count_CN.json"
        }
    }

    display_names = {
        "hunyuan": "Hunyuan",
        "kolors":  "Kolors",
        "qwen":    "Qwen",
        "wan2.1":  "Wan2.1",
    }

    # ============================
    # 2. 配色：每个模型一个颜色，EN/CN 靠背景区分
    # ============================
    model_colors = {
        "hunyuan": "#7F2704",
        "kolors":  "#D94801",
        "qwen":    "#FD8D3C",
        "wan2.1":  "#FDD0A2",
    }

    # ============================
    # 3. 读取数据
    # ============================
    proportions = {g: {m: {} for m in models} for g in genders}

    for model in models:
        for lang in languages:
            with open(paths[model][lang], "r") as f:
                data = json.load(f)
            counts = data["person"]
            total  = sum(counts.values())
            for gender in genders:
                proportions[gender][model][lang] = counts.get(gender, 0) / total

    # ============================
    # 4. x 坐标
    # ============================
    num_genders = len(genders)
    num_models  = len(models)

    bar_width         = 0.06
    model_group_width = 2 * bar_width + 0.02
    intra_gap         = 0.12
    category_gap      = 0.5

    group_width    = num_models * model_group_width + (num_models - 1) * intra_gap
    gender_centers = np.arange(num_genders) * (group_width + category_gap)

    fig, ax = plt.subplots(figsize=(8, 6))

    # ============================
    # 5. 背景色区分 EN / CN
    # ============================
    for g_idx in range(num_genders):
        center      = gender_centers[g_idx]
        group_start = center - group_width / 2.0

        for m_idx in range(num_models):
            offset       = m_idx * (model_group_width + intra_gap)
            group_center = group_start + offset + model_group_width / 2.0

            en_x = group_center - bar_width / 2.0
            cn_x = group_center + bar_width / 2.0

            ax.axvspan(en_x - bar_width * 0.6, en_x + bar_width * 0.6,
                       alpha=0.15, color="#4C8BE8", zorder=0)
            ax.axvspan(cn_x - bar_width * 0.6, cn_x + bar_width * 0.6,
                       alpha=0.15, color="#D94801", zorder=0)

    # ============================
    # 6. 画柱子
    # ============================
    for g_idx, gender in enumerate(genders):
        center      = gender_centers[g_idx]
        group_start = center - group_width / 2.0

        for m_idx, model in enumerate(models):
            offset       = m_idx * (model_group_width + intra_gap)
            group_center = group_start + offset + model_group_width / 2.0

            en_x = group_center - bar_width / 2.0
            cn_x = group_center + bar_width / 2.0

            ax.bar(en_x, proportions[gender][model]["English"],
                   width=bar_width, color=model_colors[model],
                   alpha=0.88, edgecolor="white", linewidth=0.4, zorder=3)
            ax.bar(cn_x, proportions[gender][model]["Chinese"],
                   width=bar_width, color=model_colors[model],
                   alpha=0.88, edgecolor="white", linewidth=0.4, zorder=3)

    # ============================
    # 7. 轴
    # ============================
    ax.set_xticks(gender_centers)
    ax.set_xticklabels(genders, fontsize=27)
    ax.set_ylabel("Proportion", fontsize=23)
    ax.tick_params(axis="both", labelsize=27)
    ax.set_xlim(
        gender_centers[0]  - group_width / 2.0 - bar_width * 2,
        gender_centers[-1] + group_width / 2.0 + bar_width * 2
    )

    # ============================
    # 8. 图例
    # ============================
    en_patch = mpatches.Patch(color="#4C8BE8", alpha=0.4, label="English")
    cn_patch = mpatches.Patch(color="#D94801", alpha=0.4, label="Chinese")

    model_handles = [
        mpatches.Patch(color=model_colors[m], label=display_names[m])
        for m in models
    ]

    final_handles = [en_patch, model_handles[0], model_handles[1],
                     cn_patch, model_handles[2], model_handles[3]]
    final_labels  = ["EN", display_names[models[0]], display_names[models[1]],
                     "CN", display_names[models[2]], display_names[models[3]]]

    leg = ax.legend(
        final_handles, final_labels,
        ncol=2, fontsize=15,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=0.9,
        handlelength=1.2, handleheight=1.0,
        borderpad=0.7, labelspacing=0.4, columnspacing=1.0,
    )

    for text in leg.get_texts():
        if text.get_text() in ("EN", "CN"):
            text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig("gender_distribution_models_EN_CN.png", dpi=300, bbox_inches="tight")
    plt.show()


def en_cn_age2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns

    sns.set_style("whitegrid")

    # ============================
    # 1. 配置
    # ============================
    age_groups = ["0-9", "10-19", "20-39", "40-59", "60+"]
    models     = ["hunyuan", "kolors", "qwen", "wan2.1"]
    languages  = ["English", "Chinese"]

    paths = {
        "hunyuan": {
            "English": "/Users/wmt/projects/bias/data/jsons/age/hunyuan_age_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/age/hunyuan_persons_age_count_CN.json"
        },
        "kolors": {
            "English": "/Users/wmt/projects/bias/data/jsons/age/kolors_age_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/age/kolors_persons_age_count_CN.json"
        },
        "qwen": {
            "English": "/Users/wmt/projects/bias/data/jsons/age/qwen_age_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/age/qwen_persons_age_count_CN.json"
        },
        "wan2.1": {
            "English": "/Users/wmt/projects/bias/data/jsons/age/Wan2.1_age_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/age/wan2.1_persons_age_count_CN.json"
        }
    }

    display_names = {
        "hunyuan": "Hunyuan",
        "kolors":  "Kolors",
        "qwen":    "Qwen",
        "wan2.1":  "Wan2.1",
    }

    # ============================
    # 2. 配色：每个模型一个颜色，EN/CN 靠背景区分
    # ============================
    model_colors = {
        "hunyuan": "#7F2704",
        "kolors":  "#D94801",
        "qwen":    "#FD8D3C",
        "wan2.1":  "#FDD0A2",
    }

    # ============================
    # 3. 读取数据
    # ============================
    proportions = {age: {model: {} for model in models} for age in age_groups}

    for model in models:
        for lang in languages:
            with open(paths[model][lang], "r") as f:
                data = json.load(f)
            counts = data["person"]
            total  = sum(counts.values())
            for age in age_groups:
                proportions[age][model][lang] = counts.get(age, 0) / total

    # ============================
    # 4. x 坐标
    # ============================
    num_ages   = len(age_groups)
    num_models = len(models)

    bar_width         = 0.06
    model_group_width = 2 * bar_width + 0.02
    intra_gap         = 0.12
    age_gap           = 0.5

    group_width  = num_models * model_group_width + (num_models - 1) * intra_gap
    age_centers  = np.arange(num_ages) * (group_width + age_gap)

    fig, ax = plt.subplots(figsize=(12, 6))

    # ============================
    # 5. 背景色区分 EN / CN
    # ============================
    for a_idx in range(num_ages):
        center      = age_centers[a_idx]
        group_start = center - group_width / 2.0

        for m_idx in range(num_models):
            offset       = m_idx * (model_group_width + intra_gap)
            group_center = group_start + offset + model_group_width / 2.0

            en_x = group_center - bar_width / 2.0
            cn_x = group_center + bar_width / 2.0

            ax.axvspan(en_x - bar_width * 0.6, en_x + bar_width * 0.6,
                       alpha=0.15, color="#4C8BE8", zorder=0)
            ax.axvspan(cn_x - bar_width * 0.6, cn_x + bar_width * 0.6,
                       alpha=0.15, color="#D94801", zorder=0)

    # ============================
    # 6. 画柱子
    # ============================
    for a_idx, age in enumerate(age_groups):
        center      = age_centers[a_idx]
        group_start = center - group_width / 2.0

        for m_idx, model in enumerate(models):
            offset       = m_idx * (model_group_width + intra_gap)
            group_center = group_start + offset + model_group_width / 2.0

            en_x = group_center - bar_width / 2.0
            cn_x = group_center + bar_width / 2.0

            ax.bar(en_x, proportions[age][model]["English"],
                   width=bar_width, color=model_colors[model],
                   alpha=0.88, edgecolor="white", linewidth=0.4, zorder=3)
            ax.bar(cn_x, proportions[age][model]["Chinese"],
                   width=bar_width, color=model_colors[model],
                   alpha=0.88, edgecolor="white", linewidth=0.4, zorder=3)

    # ============================
    # 7. 轴
    # ============================
    ax.set_xticks(age_centers)
    ax.set_xticklabels(age_groups, fontsize=27)
    ax.set_ylabel("Proportion", fontsize=23)
    ax.tick_params(axis="both", labelsize=27)
    ax.set_xlim(
        age_centers[0]  - group_width / 2.0 - bar_width * 2,
        age_centers[-1] + group_width / 2.0 + bar_width * 2
    )

    # ============================
    # 8. 图例
    # ============================
    en_patch = mpatches.Patch(color="#4C8BE8", alpha=0.4, label="English")
    cn_patch = mpatches.Patch(color="#D94801", alpha=0.4, label="Chinese")

    model_handles = [
        mpatches.Patch(color=model_colors[m], label=display_names[m])
        for m in models
    ]

    final_handles = [en_patch, model_handles[0], model_handles[1],
                     cn_patch, model_handles[2], model_handles[3]]
    final_labels  = ["EN", display_names[models[0]], display_names[models[1]],
                     "CN", display_names[models[2]], display_names[models[3]]]

    leg = ax.legend(
        final_handles, final_labels,
        ncol=2, fontsize=15,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=0.9,
        handlelength=1.2, handleheight=1.0,
        borderpad=0.7, labelspacing=0.4, columnspacing=1.0,
    )

    for text in leg.get_texts():
        if text.get_text() in ("EN", "CN"):
            text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig("age_distribution_models_EN_CN.png", dpi=300, bbox_inches="tight")
    plt.show()

def en_cn_gender():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # ============================
    # 1. 配置
    # ============================
    genders = ["Male", "Female"]
    models = ["hunyuan", "kolors", "qwen", "wan2.1"]
    languages = ["English", "Chinese"]

    paths = {
        "hunyuan": {
            "English": "/Users/wmt/projects/bias/data/jsons/gender/hunyuan_gender_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/gender/hunyuan_persons_gender_count_CN.json"
        },
        "kolors": {
            "English": "/Users/wmt/projects/bias/data/jsons/gender/kolors_gender_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/gender/kolors_persons_gender_count_CN.json"
        },
        "qwen": {
            "English": "/Users/wmt/projects/bias/data/jsons/gender/qwen_gender_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/gender/qwen_persons_gender_count_CN.json"
        },
        "wan2.1": {
            "English": "/Users/wmt/projects/bias/data/jsons/gender/Wan2.1_gender_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/gender/wan2.1_persons_gender_count_CN.json"
        }
    }

    display_names = {
        "hunyuan": "Hunyuan",
        "kolors": "Kolors",
        "qwen": "Qwen",
        "wan2.1": "Wan2.1",
    }

    # ============================
    # 2. 颜色（与 race EN/CN 图完全一致）
    # ============================
    english_colors = [
        "#08306B",
        "#2171B5",
        "#6BAED6",
        "#C6DBEF",
    ]

    chinese_colors = {
        "hunyuan": "#7F2704",
        "kolors": "#A63603",
        "qwen": "#E6550D",
        "wan2.1": "#FDBE85",
    }

    # ============================
    # 3. 读取数据
    # ============================
    proportions = {g: {m: {} for m in models} for g in genders}

    for model in models:
        for lang in languages:
            with open(paths[model][lang], "r") as f:
                data = json.load(f)
            counts = data["person"]
            total = sum(counts.values())
            for gender in genders:
                proportions[gender][model][lang] = counts.get(gender, 0) / total

    # ============================
    # 4. x 坐标（与 race 版本写法一致）
    # ============================
    num_genders = len(genders)
    num_models = len(models)

    bar_width = 0.06
    model_group_width = 2 * bar_width + 0.04
    category_gap = 0.4

    gender_centers = np.arange(num_genders) * (
        num_models * model_group_width + category_gap
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # ============================
    # 5. 画柱子
    # ============================
    for g_idx, gender in enumerate(genders):
        center = gender_centers[g_idx]
        total_width = num_models * model_group_width
        start = center - total_width / 2.0

        for m_idx, model in enumerate(models):
            group_center = start + m_idx * model_group_width + model_group_width / 2.0

            x_en = group_center - bar_width / 2.0
            x_cn = group_center + bar_width / 2.0

            ax.bar(
                x_en,
                proportions[gender][model]["English"],
                width=bar_width,
                color=english_colors[m_idx],
                label=f"{display_names[model]} EN"
            )
            ax.bar(
                x_cn,
                proportions[gender][model]["Chinese"],
                width=bar_width,
                color=chinese_colors[model],
                label=f"{display_names[model]} CN"
            )

    # ============================
    # 6. 轴 & legend（完全对齐 race 版本）
    # ============================
    ax.set_xticks(gender_centers)
    ax.set_xticklabels(genders, fontsize=27)
    ax.set_ylabel("Proportion", fontsize=23)

    ax.tick_params(axis="both", labelsize=27)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=17,
        handlelength=1.4,
        handleheight=0.9,
        markerscale=0.7,
        bbox_to_anchor=(0.37, 1),
        loc="upper left",
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("gender_distribution_models_EN_CN.png", dpi=300, bbox_inches="tight")
    plt.show()




def en_cn_age():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # ============================
    # 1. 配置
    # ============================
    age_groups = ["0-9", "10-19", "20-39", "40-59", "60+"]
    models = ["hunyuan", "kolors", "qwen", "wan2.1"]
    languages = ["English", "Chinese"]

    paths = {
        "hunyuan": {
            "English": "/Users/wmt/projects/bias/data/jsons/age/hunyuan_age_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/age/hunyuan_persons_age_count_CN.json"
        },
        "kolors": {
            "English": "/Users/wmt/projects/bias/data/jsons/age/kolors_age_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/age/kolors_persons_age_count_CN.json"
        },
        "qwen": {
            "English": "/Users/wmt/projects/bias/data/jsons/age/qwen_age_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/age/qwen_persons_age_count_CN.json"
        },
        "wan2.1": {
            "English": "/Users/wmt/projects/bias/data/jsons/age/Wan2.1_age_counts.json",
            "Chinese": "/Users/wmt/projects/bias/data/jsons/age/wan2.1_persons_age_count_CN.json"
        }
    }

    display_names = {
        "hunyuan": "Hunyuan",
        "kolors": "Kolors",
        "qwen": "Qwen",
        "wan2.1": "Wan2.1",
    }

    # ============================
    # 2. 颜色（与 race / gender 完全一致）
    # ============================
    english_colors = [
        "#08306B",
        "#2171B5",
        "#6BAED6",
        "#C6DBEF",
    ]

    chinese_colors = {
        "hunyuan": "#7F2704",
        "kolors": "#A63603",
        "qwen": "#E6550D",
        "wan2.1": "#FDBE85",
    }

    # ============================
    # 3. 读取数据
    # ============================
    proportions = {age: {model: {} for model in models} for age in age_groups}

    for model in models:
        for lang in languages:
            with open(paths[model][lang], "r") as f:
                data = json.load(f)

            counts = data["person"]
            total = sum(counts.values())

            for age in age_groups:
                proportions[age][model][lang] = counts.get(age, 0) / total

    # ============================
    # 4. x 坐标（与 race / gender 写法一致）
    # ============================
    num_ages = len(age_groups)
    num_models = len(models)

    bar_width = 0.06
    model_group_width = 2 * bar_width + 0.04
    age_gap = 0.4

    age_centers = np.arange(num_ages) * (
        num_models * model_group_width + age_gap
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # ============================
    # 5. 绘制柱子
    # ============================
    for a_idx, age in enumerate(age_groups):
        center = age_centers[a_idx]
        total_width = num_models * model_group_width
        start = center - total_width / 2.0

        for m_idx, model in enumerate(models):
            group_center = start + m_idx * model_group_width + model_group_width / 2.0
            x_en = group_center - bar_width / 2.0
            x_cn = group_center + bar_width / 2.0

            ax.bar(
                x_en,
                proportions[age][model]["English"],
                width=bar_width,
                color=english_colors[m_idx],
                label=f"{display_names[model]} EN"
            )
            ax.bar(
                x_cn,
                proportions[age][model]["Chinese"],
                width=bar_width,
                color=chinese_colors[model],
                label=f"{display_names[model]} CN"
            )

    # ============================
    # 6. 轴 & legend（完全对齐 race / gender）
    # ============================
    ax.set_xticks(age_centers)
    ax.set_xticklabels(age_groups, fontsize=27)
    ax.set_ylabel("Proportion", fontsize=23)

    ax.tick_params(axis="both", labelsize=27)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=18,
        handlelength=1.4,
        handleheight=0.9,
        markerscale=0.7,
        bbox_to_anchor=(0.75, 1),
        loc="upper left",
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("age_distribution_models_EN_CN.png", dpi=300, bbox_inches="tight")
    plt.show()



if __name__ == '__main__':
    # preprocess_race()
    # preprocess_gender()
    # preprocess_age()
    en_cn_race2()
    en_cn_gender2()
    en_cn_age2()