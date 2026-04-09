import pandas as pd
import numpy as np
import json
import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def neutral():
    # ============================
    # 1. 读取 CSV
    # ============================
    csv_path = "/content/bias/chinese/hunyuan_person_res.csv"
    df = pd.read_csv(csv_path)

    # ============================
    # 2. 只保留 face0 行
    # ============================
    df_face0 = df[df["face_name_align"].astype(str).str.contains("face0", na=False)].copy()

    # ============================
    # 3. Race 处理：完全按照你给的逻辑
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
        """解析 race_scores_fair"""
        if isinstance(scores, (list, tuple, np.ndarray)):
            return list(scores)

        if isinstance(scores, str):
            s = scores.strip()

            # numpy 风格 "[0.1 0.2 0.3]"
            if "[" in s and "]" in s and "," not in s:
                s_clean = s.replace("[", "").replace("]", "")
                return [float(x) for x in s_clean.split()]

            # python list 风格
            try:
                return list(ast.literal_eval(s))
            except Exception:
                raise ValueError(f"无法解析 race_scores_fair: {scores}")

        raise ValueError(f"未知格式: {scores}")

    def get_merged_race(row):
        race = row["race"]

        # Middle Eastern → 第二高得分 race
        if race == "Middle Eastern":
            scores = parse_scores(row["race_scores_fair"])
            scores_arr = np.array(scores, dtype=float)

            sorted_idx = np.argsort(scores_arr)
            idx_second = sorted_idx[-2]
            race = race_labels[idx_second]

        # 合并逻辑（与你之前保持一致）
        if race in ["Latino_Hispanic", "Indian"]:
            return "Others"
        if race in ["East Asian", "Southeast Asian"]:
            return "Asian"
        if race in ["White", "Black"]:
            return race

        return race   # 理论不会出现 Middle Eastern

    df_face0["race_merged"] = df_face0.apply(get_merged_race, axis=1)

    # ============================
    # 4. Gender 清洗
    # ============================
    mapping_gender = {
        "m": "Male", "male": "Male", "1": "Male",
        "f": "Female", "female": "Female", "0": "Female"
    }

    df_face0["gender_clean"] = df_face0["gender"].astype(str).str.lower().map(mapping_gender)

    # ============================
    # 5. Age：解析区间再映射到 Young / Middle / Old
    # ============================

    def extract_lower_bound(age_str):
        """从 '20-29' 或 '60+' 中提取左端点"""
        if pd.isna(age_str):
            return None
        nums = re.findall(r"\d+", str(age_str))
        if len(nums) == 0:
            return None
        return int(nums[0])

    df_face0["age_lower"] = df_face0["age"].apply(extract_lower_bound)

    def map_age_group(lower):
        if lower is None:
            return None
        if lower <= 39:
            return "Young"
        elif lower <= 59:
            return "Middle"
        else:
            return "Old"

    df_face0["age_group"] = df_face0["age_lower"].apply(map_age_group)

    # 去除无法分类的
    df_face0 = df_face0.dropna(subset=["race_merged", "gender_clean", "age_group"])

    # ============================
    # 6. Intersectional 统计：Race × Gender × Age
    # ============================

    intersection_counts = {}

    for _, row in df_face0.iterrows():
        key = f"{row['race_merged']}_{row['gender_clean']}_{row['age_group']}"
        intersection_counts[key] = intersection_counts.get(key, 0) + 1

    # 排序（可选）
    intersection_counts = dict(sorted(intersection_counts.items()))

    # ============================
    # 7. 写入 JSON
    # ============================
    output_json = {
        "intersectional": intersection_counts
    }

    output_path = "/content/bias/intersection/hunyuan_chinese_stats.json"
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=4)

    print("Intersectional 统计完成！")
    print(json.dumps(output_json, indent=4))




def neutral_western():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from itertools import product
    from matplotlib.ticker import PercentFormatter

    # =========================
    # 1. Load data
    # =========================
    model_files = {
        "FLUX-Schnell": "./data/res/flux/person_res.csv",
        "Proteus": "./data/res/proteus/person_res.csv",
        "SANA": "./data/res/sana/person_res.csv",
        "SD3": "./data/res/sd3/person_res.csv",
    }

    model_colors = {
        "FLUX-Schnell": "#08306B",
        "Proteus": "#2171B5",
        "SANA": "#6BAED6",
        "SD3": "#C6DBEF",
    }

    # =========================
    # 2. Fixed semantic order
    # =========================
    age_order = ["young", "middle-aged", "old"]
    gender_order = ["Male", "Female"]
    race_order = ["white", "black", "asian", "others"]

    full_combo_order = [
        f"{a}-{g}-{r}"
        for a, g, r in product(age_order, gender_order, race_order)
    ]

    # =========================
    # 3. Race + age processing
    # =========================
    race_order_raw = [
        "White", "Black", "Latino_Hispanic",
        "East Asian", "Southeast Asian",
        "Indian", "Middle Eastern"
    ]

    def parse_race_scores(x):
        x = str(x).replace("[", "").replace("]", "")
        return np.array([float(v) for v in x.split()])

    def process_race(row):
        race = row["race"]
        scores = parse_race_scores(row["race_scores_fair"])

        if race == "Middle Eastern":
            idx = np.argsort(scores)[::-1][1]
            race = race_order_raw[idx]

        if race in ["Indian", "Latino_Hispanic"]:
            return "others"
        if race in ["East Asian", "Southeast Asian"]:
            return "asian"

        return race.lower()

    young_bins = ["0-2", "3-9", "10-19", "20-29", "30-39"]
    middle_bins = ["40-49", "50-59"]
    old_bins = ["60-69", "70+"]

    def age_group(age):
        if age in young_bins:
            return "young"
        if age in middle_bins:
            return "middle-aged"
        if age in old_bins:
            return "old"
        return np.nan

    # =========================
    # 4. Process ONE model
    # =========================
    def process_one_model(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["face_name_align"].str.contains("face0", na=False)]

        df["race_new"] = df.apply(process_race, axis=1)
        df["age_group"] = df["age"].apply(age_group)
        df = df.dropna(subset=["age_group"])
        df["gender"] = df["gender"].str.capitalize()

        df["combo"] = (
            df["age_group"] + "-" +
            df["gender"] + "-" +
            df["race_new"]
        )

        obs = df["combo"].value_counts()
        combo_count = obs.reindex(full_combo_order).fillna(0).astype(int)
        combo_prob = combo_count / combo_count.sum()

        return combo_prob.values

    # =========================
    # 5. Collect results
    # =========================
    results = {}
    for model_name, path in model_files.items():
        results[model_name] = process_one_model(path)

    # =========================
    # 6. X positions（与 Chinese 对齐）
    # =========================
    bar_width = 0.18
    num_models = len(results)
    group_center_offset = bar_width * (num_models - 1) / 2

    x_positions = []
    x_labels = []

    x = -0.5
    gap_race   = 0.0
    gap_gender = 0.4
    gap_age    = 0.2

    for age in age_order:
        for gender in gender_order:
            for race in race_order:
                x_labels.append(f"{age}-{gender}-{race}")
                x_positions.append(x)
                x += 1 + gap_race
            x += gap_gender
        x += gap_age

    x_positions = np.array(x_positions)
    old_indices = [i for i, l in enumerate(x_labels) if l.startswith("old-")]

    # =========================
    # 7. Two-panel figure
    # =========================
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(26, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]}
    )

    def fmt(v):
        return "0" if np.isclose(v, 0.0) else f"{v:.1f}"

    # ---------- TOP panel ----------
    for i, (model_name, probs) in enumerate(results.items()):
        ax_top.bar(
            x_positions + i * bar_width,
            probs * 100,
            width=bar_width,
            color=model_colors[model_name],
            label=model_name
        )

    ax_top.set_ylabel("Percentage (%)", fontsize=9)
    ax_top.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax_top.tick_params(axis="y", labelsize=12)   # 👈 对齐 Chinese
    ax_top.set_title("Joint Distribution of Age, Gender, and Race", fontsize=10)
    ax_top.legend(frameon=False, ncol=num_models, fontsize=12)

    y_max = max(bar.get_height() for bar in ax_top.patches)
    ax_top.set_ylim(0, y_max * 1.15)

    for bar in ax_top.patches:
        h = bar.get_height()
        ax_top.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.5,
            fmt(h),
            ha="center",
            va="bottom",
            fontsize=6
        )

    # ---------- BOTTOM panel ----------
    for i, (model_name, probs) in enumerate(results.items()):
        ax_bottom.bar(
            x_positions[old_indices] + i * bar_width,
            probs[old_indices] * 100,
            width=bar_width,
            color=model_colors[model_name]
        )

    ax_bottom.set_ylim(0, 2.0)
    ax_bottom.set_ylabel("Percentage (%)", fontsize=8)
    ax_bottom.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax_bottom.tick_params(axis="y", labelsize=12)  # 👈 对齐 Chinese
    ax_bottom.set_title("Zoom-in: Old Age part", fontsize=9)

    for bar in ax_bottom.patches:
        h = bar.get_height()
        ax_bottom.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.04,
            fmt(h),
            ha="center",
            va="bottom",
            fontsize=5
        )

    # ---------- Shared X-axis ----------
    ax_bottom.set_xticks(x_positions + group_center_offset)
    ax_bottom.set_xticklabels(
        x_labels,
        rotation=25,
        ha="right",
        fontsize=8
    )
    # ax_bottom.set_xlabel("Age–Gender–Race Combination", fontsize=9)

    # =========================
    # 8. Final layout
    # =========================
    left_margin = x_positions[0] - bar_width
    right_margin = x_positions[-1] + bar_width * (num_models + 1)
    ax_top.set_xlim(left_margin, right_margin)

    sns.despine(ax=ax_top)
    sns.despine(ax=ax_bottom)

    plt.subplots_adjust(hspace=0.18, bottom=0.32)
    plt.savefig(
        "./figs/joint_distribution_age_gender_race_western_twopanel.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()




def neutral_chinese():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from itertools import product
    from matplotlib.ticker import PercentFormatter

    # =========================
    # 1. Load data
    # =========================
    model_files = {
        "Hunyuan": "./data/res/hunyuan/person_res.csv",
        "Kolors": "./data/res/kolors/person_res.csv",
        "Qwen": "./data/res/qwen/person_res.csv",
        "Wan2.1": "./data/res/wan2.1/person_res.csv",
    }

    model_colors = {
        "Hunyuan": "#7F2704",
        "Kolors": "#A63603",
        "Qwen": "#E6550D",
        "Wan2.1": "#FDBE85",
    }

    # =========================
    # 2. Fixed semantic order
    # =========================
    age_order = ["young", "middle-aged", "old"]
    gender_order = ["Male", "Female"]
    race_order = ["white", "black", "asian", "others"]

    full_combo_order = [
        f"{a}-{g}-{r}"
        for a, g, r in product(age_order, gender_order, race_order)
    ]

    # =========================
    # 3. Race + age processing
    # =========================
    race_order_raw = [
        "White", "Black", "Latino_Hispanic",
        "East Asian", "Southeast Asian",
        "Indian", "Middle Eastern"
    ]

    def parse_race_scores(x):
        x = str(x).replace("[", "").replace("]", "")
        return np.array([float(v) for v in x.split()])

    def process_race(row):
        race = row["race"]
        scores = parse_race_scores(row["race_scores_fair"])

        if race == "Middle Eastern":
            idx = np.argsort(scores)[::-1][1]
            race = race_order_raw[idx]

        if race in ["Indian", "Latino_Hispanic"]:
            return "others"
        if race in ["East Asian", "Southeast Asian"]:
            return "asian"

        return race.lower()

    young_bins = ["0-2", "3-9", "10-19", "20-29", "30-39"]
    middle_bins = ["40-49", "50-59"]
    old_bins = ["60-69", "70+"]

    def age_group(age):
        if age in young_bins:
            return "young"
        if age in middle_bins:
            return "middle-aged"
        if age in old_bins:
            return "old"
        return np.nan

    # =========================
    # 4. Process ONE model
    # =========================
    def process_one_model(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["face_name_align"].str.contains("face0", na=False)]

        df["race_new"] = df.apply(process_race, axis=1)
        df["age_group"] = df["age"].apply(age_group)
        df = df.dropna(subset=["age_group"])
        df["gender"] = df["gender"].str.capitalize()

        df["combo"] = (
            df["age_group"] + "-" +
            df["gender"] + "-" +
            df["race_new"]
        )

        obs = df["combo"].value_counts()
        combo_count = obs.reindex(full_combo_order).fillna(0).astype(int)
        combo_prob = combo_count / combo_count.sum()

        return combo_prob.values

    # =========================
    # 5. Collect results
    # =========================
    results = {}
    for model_name, path in model_files.items():
        results[model_name] = process_one_model(path)

    # =========================
    # 6. X positions（仅缩小 age 大类间距）
    # =========================
    bar_width = 0.18
    num_models = len(results)
    group_center_offset = bar_width * (num_models - 1) / 2

    x_positions = []
    x_labels = []

    x = -0.5
    gap_race   = 0.0
    gap_gender = 0.4
    gap_age    = 0.2   # 👈 唯一改动（原来是 1.6）

    for age in age_order:
        for gender in gender_order:
            for race in race_order:
                x_labels.append(f"{age}-{gender}-{race}")
                x_positions.append(x)
                x += 1 + gap_race
            x += gap_gender
        x += gap_age

    x_positions = np.array(x_positions)
    old_indices = [i for i, l in enumerate(x_labels) if l.startswith("old-")]

    # =========================
    # 7. Two-panel figure（保持原样）
    # =========================
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(26, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]}
    )

    def fmt(v):
        return "0" if np.isclose(v, 0.0) else f"{v:.1f}"

    # ---------- TOP panel ----------
    for i, (model_name, probs) in enumerate(results.items()):
        ax_top.bar(
            x_positions + i * bar_width,
            probs * 100,
            width=bar_width,
            color=model_colors[model_name],
            label=model_name
        )

    ax_top.set_ylabel("Percentage (%)", fontsize=9)
    ax_top.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax_top.set_title("Joint Distribution of Age, Gender, and Race", fontsize=10)
    ax_top.tick_params(axis="y", labelsize=12)  # 👈 新增
    ax_top.legend(frameon=False, ncol=num_models, fontsize=12)

    y_max = max(bar.get_height() for bar in ax_top.patches)
    ax_top.set_ylim(0, y_max * 1.15)

    for bar in ax_top.patches:
        h = bar.get_height()
        ax_top.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.5,
            fmt(h),
            ha="center",
            va="bottom",
            fontsize=6
        )

    # ---------- BOTTOM panel ----------
    for i, (model_name, probs) in enumerate(results.items()):
        ax_bottom.bar(
            x_positions[old_indices] + i * bar_width,
            probs[old_indices] * 100,
            width=bar_width,
            color=model_colors[model_name]
        )

    ax_bottom.set_ylim(0, 2.0)
    ax_bottom.set_ylabel("Percentage (%)", fontsize=8)
    ax_bottom.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax_bottom.tick_params(axis="y", labelsize=12)  # 👈 新增
    ax_bottom.set_title("Zoom-in: Old Age part", fontsize=9)

    for bar in ax_bottom.patches:
        h = bar.get_height()
        ax_bottom.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.04,
            fmt(h),
            ha="center",
            va="bottom",
            fontsize=7
        )

    # ---------- Shared X-axis ----------
    ax_bottom.set_xticks(x_positions + group_center_offset)
    ax_bottom.set_xticklabels(
        x_labels,
        rotation=25,
        ha="right",
        fontsize=8
    )
    # ax_bottom.set_xlabel("Age–Gender–Race Combination", fontsize=9)

    # =========================
    # 8. Final layout
    # =========================
    left_margin = x_positions[0] - bar_width
    right_margin = x_positions[-1] + bar_width * (num_models + 1)
    ax_top.set_xlim(left_margin, right_margin)

    sns.despine(ax=ax_top)
    sns.despine(ax=ax_bottom)

    plt.subplots_adjust(hspace=0.18, bottom=0.32)
    plt.savefig(
        "./figs/joint_distribution_age_gender_race_chinese_twopanel.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()



def neutral_chinese_zoom2():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from itertools import product
    from matplotlib.ticker import PercentFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    # =========================
    # 1. Load data
    # =========================
    model_files = {
        "Hunyuan": "/Users/wmt/projects/bias/data/corrected/hunyuan/neutral_res.csv",
        "Kolors": "/Users/wmt/projects/bias/data/corrected/kolors/neutral_res.csv",
        "Qwen": "/Users/wmt/projects/bias/data/corrected/qwen/neutral_res.csv",
        "Wan2.1": "/Users/wmt/projects/bias/data/corrected/wan2.1/neutral_res.csv",
    }

    model_colors = {
        "Hunyuan": "#7F2704",
        "Qwen": "#D94801",
        "Kolors": "#FD8D3C",
        "Wan2.1": "#FDD0A2",
    }

    # =========================
    # 2. Fixed semantic order
    # =========================
    age_order = ["young", "middle-aged", "old"]
    gender_order = ["male", "female"]
    race_order = ["white", "black", "asian", "others"]

    full_combo_order = [
        f"{a}-{g}-{r}"
        for a, g, r in product(age_order, gender_order, race_order)
    ]

    # =========================
    # 3. Race + age processing
    # =========================
    race_order_raw = [
        "White", "Black", "Latino_Hispanic",
        "East Asian", "Southeast Asian",
        "Indian", "Middle Eastern"
    ]

    def parse_race_scores(x):
        x = str(x).replace("[", "").replace("]", "")
        return np.array([float(v) for v in x.split()])

    def process_race(row):
        race = row["race"]
        scores = parse_race_scores(row["race_scores_fair"])
        if race == "Middle Eastern":
            idx = np.argsort(scores)[::-1][1]
            race = race_order_raw[idx]
        if race in ["Indian", "Latino_Hispanic"]:
            return "others"
        if race in ["East Asian", "Southeast Asian"]:
            return "asian"
        return race.lower()

    young_bins = ["0-2", "3-9", "10-19", "20-29", "30-39"]
    middle_bins = ["40-49", "50-59"]
    old_bins = ["60-69", "70+"]

    def age_group(age):
        if age in young_bins:  return "young"
        if age in middle_bins: return "middle-aged"
        if age in old_bins:    return "old"
        return np.nan

    # =========================
    # 4. Process ONE model
    # =========================
    def process_one_model(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["face_name_align"].str.contains("face0", na=False)]
        df["race_new"] = df.apply(process_race, axis=1)
        df["age_group"] = df["age"].apply(age_group)
        df = df.dropna(subset=["age_group"])
        df["gender"] = df["gender"].str.lower()
        df["combo"] = df["age_group"] + "-" + df["gender"] + "-" + df["race_new"]

        obs = df["combo"].value_counts()
        combo_count = obs.reindex(full_combo_order).fillna(0).astype(int)
        combo_prob = combo_count / combo_count.sum()
        return combo_prob.values

    # =========================
    # 5. Collect results
    # =========================
    results = {}
    for model_name, path in model_files.items():
        results[model_name] = process_one_model(path)

    # =========================
    # 6. X positions
    # =========================
    bar_width = 0.18
    num_models = len(results)
    group_center_offset = bar_width * (num_models - 1) / 2

    x_positions = []
    x_labels = []

    x = -0.5
    gap_race = 0.0
    gap_gender = 0.4
    gap_age = 0.2

    for age in age_order:
        for gender in gender_order:
            for race in race_order:
                x_labels.append(f"{age}-{gender}-{race}")
                x_positions.append(x)
                x += 1 + gap_race
            x += gap_gender
        x += gap_age

    x_positions = np.array(x_positions)
    old_indices = [i for i, l in enumerate(x_labels) if l.startswith("old-")]

    # 用于 mark_inset 的 x 范围（old 区域两端各留半格余量）
    old_x_min = x_positions[old_indices[0]] - bar_width * 1.5
    old_x_max = x_positions[old_indices[-1]] + bar_width * (num_models + 0.5)

    # =========================
    # 7. 主图 + zoom-in inset
    # =========================
    # 用 ticks 风格：去掉密集网格，只保留轴刻度
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)

    fig, ax = plt.subplots(figsize=(26, 6))

    # ---- 手动添加轻薄横向网格线（比 whitegrid 更精致）----
    ax.yaxis.grid(True, linewidth=0.4, color="#d0d0d0", alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ---------- 主图柱子 ----------
    for i, (model_name, probs) in enumerate(results.items()):
        ax.bar(
            x_positions + i * bar_width,
            probs * 100,
            width=bar_width,
            color=model_colors[model_name],
            alpha=0.88,  # 轻微透明，边界感更强
            label=model_name,
            zorder=3
        )

    ax.set_ylabel("Percentage (%)", fontsize=20)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("Chinese Models", fontsize=26, pad=14)
    ax.tick_params(axis="y", labelsize=16)

    # ---- 图例放在左上角，避开右侧 inset 区域 ----
    ax.legend(
        frameon=False,
        ncol=1,
        fontsize=16,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98)
    )

    # 自动 y 上限
    y_max = max(bar.get_height() for bar in ax.patches)
    ax.set_ylim(0, y_max * 1.15)

    # ---- 数值标注：old 区域跳过（inset 已标注），其余全标；0 横排，非零竖排 ----
    # ax.patches 的顺序是：先所有 model0 的柱，再 model1，以此类推
    # 每个 model 有 len(full_combo_order) 根柱子
    n_combos = len(full_combo_order)
    for patch_idx, bar in enumerate(ax.patches):
        combo_idx = patch_idx % n_combos  # 当前柱对应哪个 combo
        if combo_idx in old_indices:
            continue  # old 区域不标（inset 已标）
        h = bar.get_height()
        if np.isclose(h, 0.0):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.6,
                "0",
                ha="center", va="bottom",
                fontsize=13, color="black",
                rotation=0, zorder=4
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.6,
                f"{h:.1f}",
                ha="center", va="bottom",
                fontsize=13, color="black",
                rotation=90, zorder=4
            )

    # ---------- Zoom-in inset for old ----------
    axins = inset_axes(
        ax,
        width="35%",
        height="45%",
        loc="upper right",
        borderpad=1.2
    )

    for i, (model_name, probs) in enumerate(results.items()):
        axins.bar(
            x_positions[old_indices] + i * bar_width,
            probs[old_indices] * 100,
            width=bar_width,
            color=model_colors[model_name],
            alpha=0.88,
            zorder=3
        )

    # inset 也加轻薄网格
    axins.yaxis.grid(True, linewidth=0.3, color="#d0d0d0", alpha=0.8, zorder=0)
    axins.set_axisbelow(True)

    axins.set_ylim(0, 2.0)
    axins.set_title("Zoom-in: Old Age", fontsize=19, pad=6)
    axins.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axins.tick_params(axis="y", labelsize=13)
    axins.tick_params(axis="x", labelsize=10, rotation=30)

    axins.set_xticks(x_positions[old_indices] + group_center_offset)
    axins.set_xticklabels(
        [x_labels[i] for i in old_indices],
        rotation=30,
        ha="right",
        fontsize=10
    )

    # inset 数值标注：所有值都标注，0 横排，非零竖排
    for bar in axins.patches:
        h = bar.get_height()
        if np.isclose(h, 0.0):
            axins.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.04,
                "0",
                ha="center", va="bottom",
                fontsize=11, color="black",
                rotation=0, zorder=4
            )
        else:
            axins.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.04,
                f"{h:.1f}",
                ha="center", va="bottom",
                fontsize=11, color="black",
                rotation=90, zorder=4
            )

    # ---- mark_inset：连接主图 old 区域与 inset ----
    axins.set_xlim(old_x_min, old_x_max)
    mark_inset(
        ax, axins,
        loc1=2, loc2=3,  # 连接左上角和左下角
        fc="none",
        ec="0.45",
        lw=0.8,
        linestyle="--",
        zorder=5
    )

    # ---------- 主图 X 轴 ----------
    ax.set_xticks(x_positions + group_center_offset)
    ax.set_xticklabels(
        x_labels,
        rotation=25,
        ha="right",
        fontsize=13
    )

    left_margin = x_positions[0] - bar_width
    right_margin = x_positions[-1] + bar_width * (len(results) + 1)
    ax.set_xlim(left_margin, right_margin)

    sns.despine(ax=ax, left=False)

    plt.subplots_adjust(bottom=0.32)
    plt.savefig(
        "./joint_distribution_age_gender_race_chinese_inset.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

def neutral_chinese_zoom():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from itertools import product
    from matplotlib.ticker import PercentFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    # =========================
    # 1. Load data
    # =========================
    model_files = {
        "Hunyuan": "/Users/wmt/projects/bias/data/corrected/hunyuan/neutral_res.csv",
        "Kolors":  "/Users/wmt/projects/bias/data/corrected/kolors/neutral_res.csv",
        "Qwen":    "/Users/wmt/projects/bias/data/corrected/qwen/neutral_res.csv",
        "Wan2.1":  "/Users/wmt/projects/bias/data/corrected/wan2.1/neutral_res.csv",
    }

    model_colors = {
        "Hunyuan": "#7F2704",
        "Kolors":  "#A63603",
        "Qwen":    "#E6550D",
        "Wan2.1":  "#FDBE85",
    }

    # =========================
    # 2. Fixed semantic order
    # =========================
    age_order    = ["young", "middle-aged", "old"]
    gender_order = ["male", "female"]
    race_order   = ["white", "black", "asian", "others"]

    full_combo_order = [
        f"{a}-{g}-{r}"
        for a, g, r in product(age_order, gender_order, race_order)
    ]

    # =========================
    # 3. Race + age processing
    # =========================
    race_order_raw = [
        "White", "Black", "Latino_Hispanic",
        "East Asian", "Southeast Asian",
        "Indian", "Middle Eastern"
    ]

    def parse_race_scores(x):
        x = str(x).replace("[", "").replace("]", "")
        return np.array([float(v) for v in x.split()])

    def process_race(row):
        race   = row["race"]
        scores = parse_race_scores(row["race_scores_fair"])
        if race == "Middle Eastern":
            idx  = np.argsort(scores)[::-1][1]
            race = race_order_raw[idx]
        if race in ["Indian", "Latino_Hispanic"]:
            return "others"
        if race in ["East Asian", "Southeast Asian"]:
            return "asian"
        return race.lower()

    young_bins  = ["0-2", "3-9", "10-19", "20-29", "30-39"]
    middle_bins = ["40-49", "50-59"]
    old_bins    = ["60-69", "70+"]

    def age_group(age):
        if age in young_bins:  return "young"
        if age in middle_bins: return "middle-aged"
        if age in old_bins:    return "old"
        return np.nan

    # =========================
    # 4. Process ONE model
    # =========================
    def process_one_model(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["face_name_align"].str.contains("face0", na=False)]
        df["race_new"]  = df.apply(process_race, axis=1)
        df["age_group"] = df["age"].apply(age_group)
        df = df.dropna(subset=["age_group"])
        df["gender"] = df["gender"].str.lower()
        df["combo"]  = df["age_group"] + "-" + df["gender"] + "-" + df["race_new"]

        obs         = df["combo"].value_counts()
        combo_count = obs.reindex(full_combo_order).fillna(0).astype(int)
        combo_prob  = combo_count / combo_count.sum()
        return combo_prob.values

    # =========================
    # 5. Collect results
    # =========================
    results = {}
    for model_name, path in model_files.items():
        results[model_name] = process_one_model(path)

    # =========================
    # 6. X positions
    # =========================
    bar_width = 0.18
    num_models = len(results)
    group_center_offset = bar_width * (num_models - 1) / 2

    x_positions = []
    x_labels    = []

    x          = -0.5
    gap_race   = 0.0
    gap_gender = 0.4
    gap_age    = 0.2

    for age in age_order:
        for gender in gender_order:
            for race in race_order:
                x_labels.append(f"{age}-{gender}-{race}")
                x_positions.append(x)
                x += 1 + gap_race
            x += gap_gender
        x += gap_age

    x_positions = np.array(x_positions)
    old_indices = [i for i, l in enumerate(x_labels) if l.startswith("old-")]

    # 用于 mark_inset 的 x 范围（old 区域两端各留半格余量）
    old_x_min = x_positions[old_indices[0]]  - bar_width * 1.5
    old_x_max = x_positions[old_indices[-1]] + bar_width * (num_models + 0.5)

    # =========================
    # 7. 主图 + zoom-in inset
    # =========================
    # 用 ticks 风格：去掉密集网格，只保留轴刻度
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)

    fig, ax = plt.subplots(figsize=(26, 6))

    # ---- 手动添加轻薄横向网格线（比 whitegrid 更精致）----
    ax.yaxis.grid(True, linewidth=0.4, color="#d0d0d0", alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ---------- 主图柱子 ----------
    for i, (model_name, probs) in enumerate(results.items()):
        ax.bar(
            x_positions + i * bar_width,
            probs * 100,
            width=bar_width,
            color=model_colors[model_name],
            alpha=0.88,          # 轻微透明，边界感更强
            label=model_name,
            zorder=3
        )

    ax.set_ylabel("Percentage (%)", fontsize=20)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("Chinese Models", fontsize=26, pad=14)
    ax.tick_params(axis="y", labelsize=16)

    # ---- 图例放回图内右上角（避开 inset 区域，放在中上方）----
    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=16,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90)
    )

    # 自动 y 上限
    y_max = max(bar.get_height() for bar in ax.patches)
    ax.set_ylim(0, y_max * 1.15)

    # ---- 数值标注：0 横排，其余竖排；所有非零值都标注 ----
    for bar in ax.patches:
        h = bar.get_height()
        if np.isclose(h, 0.0):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.6,
                "0",
                ha="center", va="bottom",
                fontsize=13, color="black",
                rotation=0, zorder=4
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.6,
                f"{h:.1f}",
                ha="center", va="bottom",
                fontsize=13, color="black",
                rotation=90, zorder=4
            )

    # ---------- Zoom-in inset for old ----------
    axins = inset_axes(
        ax,
        width="35%",
        height="45%",
        loc="upper right",
        borderpad=1.2
    )

    for i, (model_name, probs) in enumerate(results.items()):
        axins.bar(
            x_positions[old_indices] + i * bar_width,
            probs[old_indices] * 100,
            width=bar_width,
            color=model_colors[model_name],
            alpha=0.88,
            zorder=3
        )

    # inset 也加轻薄网格
    axins.yaxis.grid(True, linewidth=0.3, color="#d0d0d0", alpha=0.8, zorder=0)
    axins.set_axisbelow(True)

    axins.set_ylim(0, 2.0)
    axins.set_title("Zoom-in: Old Age", fontsize=19, pad=6)
    axins.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axins.tick_params(axis="y", labelsize=13)
    axins.tick_params(axis="x", labelsize=10, rotation=30)

    axins.set_xticks(x_positions[old_indices] + group_center_offset)
    axins.set_xticklabels(
        [x_labels[i] for i in old_indices],
        rotation=30,
        ha="right",
        fontsize=10
    )

    # inset 数值标注（阈值可以低一点，因为 old 整体占比小）
    INSET_THRESHOLD = 0.3
    for bar in axins.patches:
        h = bar.get_height()
        if h < INSET_THRESHOLD:
            continue
        axins.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.06,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color="black",
            rotation=90,
            zorder=4
        )

    # ---- mark_inset：连接主图 old 区域与 inset ----
    axins.set_xlim(old_x_min, old_x_max)
    mark_inset(
        ax, axins,
        loc1=2, loc2=3,       # 连接左上角和左下角
        fc="none",
        ec="0.45",
        lw=0.8,
        linestyle="--",
        zorder=5
    )

    # ---------- 主图 X 轴 ----------
    ax.set_xticks(x_positions + group_center_offset)
    ax.set_xticklabels(
        x_labels,
        rotation=25,
        ha="right",
        fontsize=13
    )

    left_margin  = x_positions[0]  - bar_width
    right_margin = x_positions[-1] + bar_width * (len(results) + 1)
    ax.set_xlim(left_margin, right_margin)

    sns.despine(ax=ax, left=False)

    plt.subplots_adjust(bottom=0.32)
    plt.savefig(
        "./joint_distribution_age_gender_race_chinese_inset.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

def neutral_western_zoom2():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from itertools import product
    from matplotlib.ticker import PercentFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    # =========================
    # 1. Load data
    # =========================
    model_files = {
        "FLUX":    "/Users/wmt/projects/bias/data/corrected/flux/neutral_res.csv",
        "Proteus": "/Users/wmt/projects/bias/data/corrected/proteus/neutral_res.csv",
        "SANA":    "/Users/wmt/projects/bias/data/corrected/sana/neutral_res.csv",
        "SD3":     "/Users/wmt/projects/bias/data/corrected/sd3/neutral_res.csv",
    }

    model_colors = {
        "FLUX":    "#08306B",
        "Proteus": "#2171B5",
        "SANA":    "#BDD7E7",
        "SD3":     "#6BAED6",
    }

    # =========================
    # 2. Fixed semantic order
    # =========================
    age_order    = ["young", "middle-aged", "old"]
    gender_order = ["male", "female"]
    race_order   = ["white", "black", "asian", "others"]

    full_combo_order = [
        f"{a}-{g}-{r}"
        for a, g, r in product(age_order, gender_order, race_order)
    ]

    # =========================
    # 3. Race + age processing
    # =========================
    race_order_raw = [
        "White", "Black", "Latino_Hispanic",
        "East Asian", "Southeast Asian",
        "Indian", "Middle Eastern"
    ]

    def parse_race_scores(x):
        x = str(x).replace("[", "").replace("]", "")
        return np.array([float(v) for v in x.split()])

    def process_race(row):
        race   = row["race"]
        scores = parse_race_scores(row["race_scores_fair"])
        if race == "Middle Eastern":
            idx  = np.argsort(scores)[::-1][1]
            race = race_order_raw[idx]
        if race in ["Indian", "Latino_Hispanic"]:
            return "others"
        if race in ["East Asian", "Southeast Asian"]:
            return "asian"
        return race.lower()

    young_bins  = ["0-2", "3-9", "10-19", "20-29", "30-39"]
    middle_bins = ["40-49", "50-59"]
    old_bins    = ["60-69", "70+"]

    def age_group(age):
        if age in young_bins:  return "young"
        if age in middle_bins: return "middle-aged"
        if age in old_bins:    return "old"
        return np.nan

    # =========================
    # 4. Process ONE model
    # =========================
    def process_one_model(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["face_name_align"].str.contains("face0", na=False)]
        df["race_new"]  = df.apply(process_race, axis=1)
        df["age_group"] = df["age"].apply(age_group)
        df = df.dropna(subset=["age_group"])
        df["gender"] = df["gender"].str.lower()
        df["combo"]  = df["age_group"] + "-" + df["gender"] + "-" + df["race_new"]

        obs         = df["combo"].value_counts()
        combo_count = obs.reindex(full_combo_order).fillna(0).astype(int)
        combo_prob  = combo_count / combo_count.sum()
        return combo_prob.values

    # =========================
    # 5. Collect results
    # =========================
    results = {}
    for model_name, path in model_files.items():
        results[model_name] = process_one_model(path)

    # =========================
    # 6. X positions
    # =========================
    bar_width = 0.18
    num_models = len(results)
    group_center_offset = bar_width * (num_models - 1) / 2

    x_positions = []
    x_labels    = []

    x          = -0.5
    gap_race   = 0.0
    gap_gender = 0.4
    gap_age    = 0.2

    for age in age_order:
        for gender in gender_order:
            for race in race_order:
                x_labels.append(f"{age}-{gender}-{race}")
                x_positions.append(x)
                x += 1 + gap_race
            x += gap_gender
        x += gap_age

    x_positions = np.array(x_positions)
    old_indices = [i for i, l in enumerate(x_labels) if l.startswith("old-")]

    # 用于 mark_inset 的 x 范围
    old_x_min = x_positions[old_indices[0]]  - bar_width * 1.5
    old_x_max = x_positions[old_indices[-1]] + bar_width * (num_models + 0.5)

    # =========================
    # 7. 主图 + zoom-in inset
    # =========================
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)

    fig, ax = plt.subplots(figsize=(26, 6))

    # 轻薄横向网格线
    ax.yaxis.grid(True, linewidth=0.4, color="#d0d0d0", alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ---------- 主图柱子 ----------
    for i, (model_name, probs) in enumerate(results.items()):
        ax.bar(
            x_positions + i * bar_width,
            probs * 100,
            width=bar_width,
            color=model_colors[model_name],
            alpha=0.88,
            label=model_name,
            zorder=3
        )

    ax.set_ylabel("Percentage (%)", fontsize=20)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("Western Models", fontsize=26, pad=14)
    ax.tick_params(axis="y", labelsize=16)

    # 图例放在左上角，避开右侧 inset
    ax.legend(
        frameon=False,
        ncol=1,
        fontsize=16,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98)
    )

    # 自动 y 上限
    y_max = max(bar.get_height() for bar in ax.patches)
    ax.set_ylim(0, y_max * 1.15)

    # 数值标注：old 区域跳过（inset 已标），其余全标；0 横排，非零竖排
    n_combos = len(full_combo_order)
    for patch_idx, bar in enumerate(ax.patches):
        combo_idx = patch_idx % n_combos
        if combo_idx in old_indices:
            continue
        h = bar.get_height()
        if np.isclose(h, 0.0):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.6,
                "0",
                ha="center", va="bottom",
                fontsize=13, color="black",
                rotation=0, zorder=4
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.6,
                f"{h:.1f}",
                ha="center", va="bottom",
                fontsize=13, color="black",
                rotation=90, zorder=4
            )

    # ---------- Zoom-in inset for old ----------
    axins = inset_axes(
        ax,
        width="35%",
        height="45%",
        loc="upper right",
        borderpad=1.2
    )

    for i, (model_name, probs) in enumerate(results.items()):
        axins.bar(
            x_positions[old_indices] + i * bar_width,
            probs[old_indices] * 100,
            width=bar_width,
            color=model_colors[model_name],
            alpha=0.88,
            zorder=3
        )

    # inset 轻薄网格
    axins.yaxis.grid(True, linewidth=0.3, color="#d0d0d0", alpha=0.8, zorder=0)
    axins.set_axisbelow(True)

    axins.set_ylim(0, 2.0)
    axins.set_title("Zoom-in: Old Age", fontsize=19, pad=6)
    axins.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axins.tick_params(axis="y", labelsize=13)
    axins.tick_params(axis="x", labelsize=10, rotation=30)

    axins.set_xticks(x_positions[old_indices] + group_center_offset)
    axins.set_xticklabels(
        [x_labels[i] for i in old_indices],
        rotation=30,
        ha="right",
        fontsize=10
    )

    # inset 数值标注：0 横排，非零竖排
    for bar in axins.patches:
        h = bar.get_height()
        if np.isclose(h, 0.0):
            axins.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.04,
                "0",
                ha="center", va="bottom",
                fontsize=11, color="black",
                rotation=0, zorder=4
            )
        else:
            axins.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.04,
                f"{h:.1f}",
                ha="center", va="bottom",
                fontsize=11, color="black",
                rotation=90, zorder=4
            )

    # mark_inset：连接主图 old 区域与 inset
    axins.set_xlim(old_x_min, old_x_max)
    mark_inset(
        ax, axins,
        loc1=2, loc2=3,
        fc="none",
        ec="0.45",
        lw=0.8,
        linestyle="--",
        zorder=5
    )

    # ---------- 主图 X 轴 ----------
    ax.set_xticks(x_positions + group_center_offset)
    ax.set_xticklabels(
        x_labels,
        rotation=25,
        ha="right",
        fontsize=13
    )

    left_margin  = x_positions[0]  - bar_width
    right_margin = x_positions[-1] + bar_width * (len(results) + 1)
    ax.set_xlim(left_margin, right_margin)

    sns.despine(ax=ax, left=False)

    plt.subplots_adjust(bottom=0.32)
    plt.savefig(
        "./joint_distribution_age_gender_race_western_inset.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

def neutral_western_zoom():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from itertools import product
    from matplotlib.ticker import PercentFormatter

    # =========================
    # 1. Load data
    # =========================
    model_files = {
        "FLUX": "/Users/wmt/projects/bias/data/corrected/flux/neutral_res.csv",
        "Proteus": "/Users/wmt/projects/bias/data/corrected/proteus/neutral_res.csv",
        "SANA": "/Users/wmt/projects/bias/data/corrected/sana/neutral_res.csv",
        "SD3": "/Users/wmt/projects/bias/data/corrected/sd3/neutral_res.csv",
    }

    model_colors = {
        "FLUX": "#08306B",
        "Proteus": "#2171B5",
        "SANA": "#6BAED6",
        "SD3": "#C6DBEF",
    }

    # =========================
    # 2. Fixed semantic order
    # =========================
    age_order = ["young", "middle-aged", "old"]
    gender_order = ["male", "female"]
    race_order = ["white", "black", "asian", "others"]

    full_combo_order = [
        f"{a}-{g}-{r}"
        for a, g, r in product(age_order, gender_order, race_order)
    ]

    # =========================
    # 3. Race + age processing
    # =========================
    race_order_raw = [
        "White", "Black", "Latino_Hispanic",
        "East Asian", "Southeast Asian",
        "Indian", "Middle Eastern"
    ]

    def parse_race_scores(x):
        x = str(x).replace("[", "").replace("]", "")
        return np.array([float(v) for v in x.split()])

    def process_race(row):
        race = row["race"]
        scores = parse_race_scores(row["race_scores_fair"])

        if race == "Middle Eastern":
            idx = np.argsort(scores)[::-1][1]
            race = race_order_raw[idx]

        if race in ["Indian", "Latino_Hispanic"]:
            return "others"
        if race in ["East Asian", "Southeast Asian"]:
            return "asian"

        return race.lower()

    young_bins = ["0-2", "3-9", "10-19", "20-29", "30-39"]
    middle_bins = ["40-49", "50-59"]
    old_bins = ["60-69", "70+"]

    def age_group(age):
        if age in young_bins:
            return "young"
        if age in middle_bins:
            return "middle-aged"
        if age in old_bins:
            return "old"
        return np.nan

    # =========================
    # 4. Process ONE model
    # =========================
    def process_one_model(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["face_name_align"].str.contains("face0", na=False)]

        df["race_new"] = df.apply(process_race, axis=1)
        df["age_group"] = df["age"].apply(age_group)
        df = df.dropna(subset=["age_group"])
        df["gender"] = df["gender"].str.lower()

        df["combo"] = (
            df["age_group"] + "-" +
            df["gender"] + "-" +
            df["race_new"]
        )

        obs = df["combo"].value_counts()
        combo_count = obs.reindex(full_combo_order).fillna(0).astype(int)
        combo_prob = combo_count / combo_count.sum()

        return combo_prob.values

    # =========================
    # 5. Collect results
    # =========================
    results = {}
    for model_name, path in model_files.items():
        results[model_name] = process_one_model(path)

    # =========================
    # 6. X positions（与 Chinese 对齐）
    # =========================
    bar_width = 0.18
    num_models = len(results)
    group_center_offset = bar_width * (num_models - 1) / 2

    x_positions = []
    x_labels = []

    x = -0.5
    gap_race   = 0.0
    gap_gender = 0.4
    gap_age    = 0.2

    for age in age_order:
        for gender in gender_order:
            for race in race_order:
                x_labels.append(f"{age}-{gender}-{race}")
                x_positions.append(x)
                x += 1 + gap_race
            x += gap_gender
        x += gap_age

    x_positions = np.array(x_positions)
    old_indices = [i for i, l in enumerate(x_labels) if l.startswith("old-")]

    # =========================
    # 7. Single-panel + zoom-in inset
    # =========================
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

    fig, ax = plt.subplots(figsize=(26, 6))

    def fmt(v):
        return "0" if np.isclose(v, 0.0) else f"{v:.1f}"

    # ---------- Main panel ----------
    for i, (model_name, probs) in enumerate(results.items()):
        ax.bar(
            x_positions + i * bar_width,
            probs * 100,
            width=bar_width,
            color=model_colors[model_name],
            label=model_name
        )

    ax.set_ylabel("Percentage (%)", fontsize=20)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("Western Models", fontsize=26)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=16,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9)
    )

    # 自动 y 上限
    y_max = max(bar.get_height() for bar in ax.patches)
    ax.set_ylim(0, y_max * 1.15)

    # 数值标注（0 → 横向，其余竖排，字号统一）
    for bar in ax.patches:
        h = bar.get_height()

        if np.isclose(h, 0.0):
            label = "0"
            rotation = 0  # 横向
            color = "black"
        else:
            label = f"{h:.1f}"
            rotation = 90  # 竖排
            color = "black"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.6,
            label,
            ha="center",
            va="bottom",
            fontsize=13,  # 与 Chinese 统一
            color=color,
            rotation=rotation
        )

    # ---------- Zoom-in inset for old ----------
    axins = inset_axes(
        ax,
        width="35%",  # 放大框宽度（相对主图）
        height="40%",  # 放大框高度
        loc="upper right",
        borderpad=1.2
    )

    for i, (model_name, probs) in enumerate(results.items()):
        axins.bar(
            x_positions[old_indices] + i * bar_width,
            probs[old_indices] * 100,
            width=bar_width,
            color=model_colors[model_name]
        )

    axins.set_ylim(0, 2.0)
    axins.set_title("Zoom-in: Old Age", fontsize=19)
    axins.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axins.tick_params(axis="y", labelsize=13)
    axins.tick_params(axis="x", labelsize=13, rotation=30)

    # inset 的 x 轴标签
    axins.set_xticks(x_positions[old_indices] + group_center_offset)
    axins.set_xticklabels(
        [x_labels[i] for i in old_indices],
        rotation=30,
        ha="right",
        fontsize=10
    )

    # inset 数值（0 → 横向，其余竖排，字号统一）
    for bar in axins.patches:
        h = bar.get_height()

        if np.isclose(h, 0.0):
            label = "0"
            rotation = 0
            color = "black"
        else:
            label = f"{h:.1f}"
            rotation = 90
            color = "black"

        axins.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.08,
            label,
            ha="center",
            va="bottom",
            fontsize=13,  # 与 Chinese 统一
            color=color,
            rotation=rotation
        )

    # ---------- Shared X-axis ----------
    ax.set_xticks(x_positions + group_center_offset)
    ax.set_xticklabels(
        x_labels,
        rotation=25,
        ha="right",
        fontsize=13
    )

    # =========================
    # 8. Final layout
    # =========================
    left_margin = x_positions[0] - bar_width
    right_margin = x_positions[-1] + bar_width * (len(results) + 1)
    ax.set_xlim(left_margin, right_margin)

    sns.despine(ax=ax)

    plt.subplots_adjust(bottom=0.32)
    plt.savefig(
        "./joint_distribution_age_gender_race_western_inset.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


if __name__ == '__main__':
    # neutral_chinese()
    # neutral_western()
    # neutral_chinese_zoom2()
    # neutral_western_zoom()
    neutral_western_zoom2()