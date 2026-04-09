import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

def gender_distribution():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # --------------------------------
    # 1. Models & groups
    # --------------------------------
    western_models = ["flux", "Proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "Wan2.1"]
    model_names = western_models + chinese_models

    legend_name_map = {
        "flux": "Flux",
        "Proteus": "Proteus",
        "sd3": "SD3",
        "sana": "SANA",
        "hunyuan": "Hunyuan",
        "qwen": "Qwen",
        "kolors": "Kolors",
        "Wan2.1": "Wan2.1",
    }

    # --------------------------------
    # 2. Colors
    # --------------------------------
    model_colors = {
        "flux": "#08306B",
        "Proteus": "#2171B5",
        "sd3": "#C6DBEF",
        "sana": "#6BAED6",
        "hunyuan": "#7F2704",
        "kolors": "#A63603",
        "qwen": "#E6550D",
        "Wan2.1": "#FDBE85",
    }

    # --------------------------------
    # 3. Load data
    # --------------------------------
    base_path = "/Users/wmt/projects/bias/data/jsons/gender"
    data = {}

    for model in model_names:
        with open(os.path.join(base_path, f"{model}_gender_counts.json"), "r") as f:
            data[model] = json.load(f)["person"]

    gender_groups = list(data[model_names[0]].keys())
    n_gender = len(gender_groups)

    # --------------------------------
    # 4. Convert to proportions
    # --------------------------------
    proportions = {}
    for model, counts in data.items():
        total = sum(counts.values())
        proportions[model] = [counts[g] / total for g in gender_groups]

    # --------------------------------
    # 5. Real-world gender distributions
    # --------------------------------
    real_gender_us_eu = {"Male": 0.492, "Female": 0.508}
    real_gender_china = {"Male": 0.512, "Female": 0.488}

    # --------------------------------
    # 6. Layout parameters
    # --------------------------------
    bar_width = 0.09
    intra_group_gap = bar_width * 1.7
    inter_block_gap = bar_width * 5.0

    block_width = bar_width * 8 + intra_group_gap
    step = block_width + inter_block_gap
    x = np.arange(n_gender) * step

    def get_x_pos(gender_idx, model):
        base = x[gender_idx]
        if model in western_models:
            i = western_models.index(model)
            return base + i * bar_width
        else:
            i = chinese_models.index(model)
            return base + len(western_models) * bar_width + intra_group_gap + i * bar_width

    western_block_width = len(western_models) * bar_width
    chinese_block_width = len(chinese_models) * bar_width

    def western_block_range(gender_idx):
        return x[gender_idx], x[gender_idx] + western_block_width

    def chinese_block_range(gender_idx):
        start = x[gender_idx] + western_block_width + intra_group_gap
        return start, start + chinese_block_width

    # --------------------------------
    # 7. Plot
    # --------------------------------
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Bars
    for model in model_names:
        positions = [get_x_pos(i, model) for i in range(n_gender)]
        ax.bar(
            positions,
            proportions[model],
            width=bar_width,
            color=model_colors[model],
            label=model,
            align="center"
        )

    # Real-world dashed baselines
    for i, gender in enumerate(gender_groups):

        # US + Europe
        y = real_gender_us_eu[gender]
        xmin, xmax = western_block_range(i)
        ax.hlines(
            y=y,
            xmin=xmin - bar_width / 2,
            xmax=xmax - bar_width / 2,
            linestyle="--",
            color="#4C72B0",
            linewidth=2.8,  # 👈 baseline 更粗
            label="Real-world (US+Europe)" if i == 0 else None
        )

        # China
        y = real_gender_china[gender]
        xmin, xmax = chinese_block_range(i)
        ax.hlines(
            y=y,
            xmin=xmin - bar_width / 2,
            xmax=xmax - bar_width / 2,
            linestyle="--",
            color="#C44E52",
            linewidth=2.8,  # 👈 baseline 更粗
            label="Real-world (China)" if i == 0 else None
        )

    # ↑ / ↓ arrows
    for model in model_names:
        for i, gender in enumerate(gender_groups):
            val = proportions[model][i]
            rv = real_gender_us_eu[gender] if model in western_models else real_gender_china[gender]
            xpos = get_x_pos(i, model)
            ax.annotate(
                "↑" if val > rv else "↓",
                (xpos, val + 0.015),
                ha="center",
                fontsize=18,   # 👈 箭头也放大
                color="#1e8449" if val > rv else "#6D2E46"
            )

    # --------------------------------
    # 8. Axes（统一 17 号）
    # --------------------------------
    ax.set_xticks(x + block_width / 2)
    ax.set_xticklabels(gender_groups, fontsize=27)
    ax.set_ylabel("Proportion", fontsize=23)
    ax.tick_params(axis="both", labelsize=27)

    # --------------------------------
    # 9. Legends（放大字体 + 图标）
    # --------------------------------
    handles, labels = ax.get_legend_handles_labels()

    def H(lbl):
        return handles[labels.index(lbl)]

    # Left legend: Western
    western_handles = [
        H("flux"),
        H("Proteus"),
        H("sd3"),
        H("sana"),
        H("Real-world (US+Europe)")
    ]
    western_labels = [
        legend_name_map["flux"],
        legend_name_map["Proteus"],
        legend_name_map["sd3"],
        legend_name_map["sana"],
        "Real-world (US+Europe)"
    ]

    leg_left = ax.legend(
        western_handles,
        western_labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 1),
        fontsize=12,        # 👈 legend 字体
        framealpha=0.9,
        handlelength=1.2,   # 👈 图标更大
        handleheight=1.0
    )
    ax.add_artist(leg_left)

    # Right legend: Chinese
    chinese_handles = [
        H("hunyuan"),
        H("qwen"),
        H("kolors"),
        H("Wan2.1"),
        H("Real-world (China)")
    ]
    chinese_labels = [
        legend_name_map["hunyuan"],
        legend_name_map["qwen"],
        legend_name_map["kolors"],
        legend_name_map["Wan2.1"],
        "Real-world (China)"
    ]

    ax.legend(
        chinese_handles,
        chinese_labels,
        loc="upper center",
        bbox_to_anchor=(0.526, 0.75),
        fontsize=12,  # 👈 legend 字体
        framealpha=0.9,
        handlelength=1.2,  # 👈 图标更大
        handleheight=1.0
    )

    plt.tight_layout()
    plt.savefig("neutral_gender_distribution_final.png", dpi=300, bbox_inches="tight")
    plt.show()


def gender_distribution2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # --------------------------------
    # 1. Models & groups
    # --------------------------------
    western_models = ["flux", "Proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "Wan2.1"]
    model_names = western_models + chinese_models

    legend_name_map = {
        "flux": "Flux",
        "Proteus": "Proteus",
        "sd3": "SD3",
        "sana": "SANA",
        "hunyuan": "Hunyuan",
        "qwen": "Qwen",
        "kolors": "Kolors",
        "Wan2.1": "Wan2.1",
    }

    # --------------------------------
    # 2. Colors（新配色，key 大小写对齐原版）
    # --------------------------------
    model_colors = {
        "flux": "#08306B",
        "Proteus": "#2171B5",
        "sd3": "#6BAED6",
        "sana": "#BDD7E7",
        "hunyuan": "#7F2704",
        "qwen": "#D94801",
        "kolors": "#FD8D3C",
        "Wan2.1": "#FDD0A2",
    }

    # --------------------------------
    # 3. Load data
    # --------------------------------
    base_path = "/Users/wmt/projects/bias/data/jsons/gender"
    data = {}

    for model in model_names:
        with open(os.path.join(base_path, f"{model}_gender_counts.json"), "r") as f:
            data[model] = json.load(f)["person"]

    gender_groups = list(data[model_names[0]].keys())
    n_gender = len(gender_groups)

    # --------------------------------
    # 4. Convert to proportions
    # --------------------------------
    proportions = {}
    for model, counts in data.items():
        total = sum(counts.values())
        proportions[model] = [counts[g] / total for g in gender_groups]

    # --------------------------------
    # 5. Real-world gender distributions
    # --------------------------------
    real_gender_us_eu = {"Male": 0.492, "Female": 0.508}
    real_gender_china = {"Male": 0.512, "Female": 0.488}

    # --------------------------------
    # 6. Layout parameters
    # --------------------------------
    bar_width = 0.09
    intra_group_gap = bar_width * 1.7
    inter_block_gap = bar_width * 5.0

    block_width = bar_width * 8 + intra_group_gap
    step = block_width + inter_block_gap
    x = np.arange(n_gender) * step

    def get_x_pos(gender_idx, model):
        base = x[gender_idx]
        if model in western_models:
            i = western_models.index(model)
            return base + i * bar_width
        else:
            i = chinese_models.index(model)
            return base + len(western_models) * bar_width + intra_group_gap + i * bar_width

    western_block_width = len(western_models) * bar_width
    chinese_block_width = len(chinese_models) * bar_width

    def western_block_range(gender_idx):
        return x[gender_idx], x[gender_idx] + western_block_width

    def chinese_block_range(gender_idx):
        start = x[gender_idx] + western_block_width + intra_group_gap
        return start, start + chinese_block_width

    # --------------------------------
    # 7. Plot
    # --------------------------------
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 背景色区分 Western / Chinese 区块
    for i in range(n_gender):
        western_start = x[i] - bar_width * 0.5
        western_end = x[i] + western_block_width
        chinese_start = western_end + intra_group_gap * 0.3
        chinese_end = x[i] + block_width - bar_width * 0.5
        ax.axvspan(western_start, western_end, alpha=0.04, color="#4C8BE8", zorder=0)
        ax.axvspan(chinese_start, chinese_end, alpha=0.04, color="#D94801", zorder=0)

    # Bars
    for model in model_names:
        positions = [get_x_pos(i, model) for i in range(n_gender)]
        ax.bar(
            positions,
            proportions[model],
            width=bar_width,
            color=model_colors[model],
            label=model,
            align="center"
        )

    # Real-world dashed baselines
    for i, gender in enumerate(gender_groups):
        # US + Europe
        y = real_gender_us_eu[gender]
        xmin, xmax = western_block_range(i)
        ax.hlines(
            y=y,
            xmin=xmin - bar_width / 2,
            xmax=xmax - bar_width / 2,
            linestyle="--",
            color="#4C72B0",
            linewidth=2.8,
            label="Real-world (US+Europe)" if i == 0 else None
        )

        # China
        y = real_gender_china[gender]
        xmin, xmax = chinese_block_range(i)
        ax.hlines(
            y=y,
            xmin=xmin - bar_width / 2,
            xmax=xmax - bar_width / 2,
            linestyle="--",
            color="#C44E52",
            linewidth=2.8,
            label="Real-world (China)" if i == 0 else None
        )

    # --------------------------------
    # 8. Axes
    # --------------------------------
    ax.set_xticks(x + block_width / 2)
    ax.set_xticklabels(gender_groups, fontsize=27)
    ax.set_ylabel("Proportion", fontsize=23)
    ax.tick_params(axis="both", labelsize=27)

    # --------------------------------
    # 9. Legends
    # --------------------------------
    # handles, labels = ax.get_legend_handles_labels()
    #
    # def H(lbl):
    #     return handles[labels.index(lbl)]
    #
    # # Left legend: Western
    # western_handles = [
    #     H("flux"),
    #     H("Proteus"),
    #     H("sd3"),
    #     H("sana"),
    #     H("Real-world (US+Europe)")
    # ]
    # western_labels = [
    #     legend_name_map["flux"],
    #     legend_name_map["Proteus"],
    #     legend_name_map["sd3"],
    #     legend_name_map["sana"],
    #     "Real-world (US+Europe)"
    # ]
    #
    # leg_left = ax.legend(
    #     western_handles,
    #     western_labels,
    #     loc="upper center",
    #     bbox_to_anchor=(0.55, 1),
    #     fontsize=12,
    #     framealpha=0.9,
    #     handlelength=1.2,
    #     handleheight=1.0
    # )
    # ax.add_artist(leg_left)
    #
    # # Right legend: Chinese
    # chinese_handles = [
    #     H("hunyuan"),
    #     H("qwen"),
    #     H("kolors"),
    #     H("Wan2.1"),
    #     H("Real-world (China)")
    # ]
    # chinese_labels = [
    #     legend_name_map["hunyuan"],
    #     legend_name_map["qwen"],
    #     legend_name_map["kolors"],
    #     legend_name_map["Wan2.1"],
    #     "Real-world (China)"
    # ]
    #
    # ax.legend(
    #     chinese_handles,
    #     chinese_labels,
    #     loc="upper center",
    #     bbox_to_anchor=(0.526, 0.75),
    #     fontsize=12,
    #     framealpha=0.9,
    #     handlelength=1.2,
    #     handleheight=1.0
    # )

    plt.tight_layout()
    plt.savefig("neutral_gender_distribution_final.png", dpi=300, bbox_inches="tight")
    plt.show()


def race_distribution2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # --------------------------------
    # 1. Models & groups
    # --------------------------------
    western_models = ["flux", "Proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "Wan2.1"]
    model_names = western_models + chinese_models

    legend_name_map = {
        "flux":    "Flux",
        "Proteus": "Proteus",
        "sd3":     "SD3",
        "sana":    "SANA",
        "hunyuan": "Hunyuan",
        "qwen":    "Qwen",
        "kolors":  "Kolors",
        "Wan2.1":  "Wan2.1",
    }

    # --------------------------------
    # 2. Colors（与 gender 图一致）
    # --------------------------------
    model_colors = {
        "flux":    "#08306B",
        "Proteus": "#2171B5",
        "sd3":     "#6BAED6",
        "sana":    "#BDD7E7",
        "hunyuan": "#7F2704",
        "qwen":    "#D94801",
        "kolors":  "#FD8D3C",
        "Wan2.1":  "#FDD0A2",
    }

    # --------------------------------
    # 3. Load JSON
    # --------------------------------
    base_path = "/Users/wmt/projects/bias/data/jsons/race"
    data = {}

    for model in model_names:
        with open(os.path.join(base_path, f"{model}_emotion_race_counts.json"), "r") as f:
            data[model] = json.load(f)["person"]

    race_groups = list(data[model_names[0]].keys())
    n_race = len(race_groups)

    # --------------------------------
    # 4. Convert to proportions
    # --------------------------------
    proportions = {}
    for model, counts in data.items():
        total = sum(counts.values())
        proportions[model] = [counts[r] / total for r in race_groups]

    # --------------------------------
    # 5. Real-world race distributions
    # --------------------------------
    real_race_us_eu = {
        "White":  0.76,
        "Black":  0.08,
        "Asian":  0.06,
        "Others": 0.10,
    }

    real_race_china = {
        "White":  0.01,
        "Black":  0.01,
        "Asian":  0.92,
        "Others": 0.06,
    }

    # --------------------------------
    # 6. Layout parameters
    # --------------------------------
    bar_width = 0.09
    intra_group_gap = bar_width * 1.7
    inter_block_gap = bar_width * 5.0

    block_width = bar_width * 8 + intra_group_gap
    step = block_width + inter_block_gap
    x = np.arange(n_race) * step

    def get_x_pos(race_idx, model):
        base = x[race_idx]
        if model in western_models:
            i = western_models.index(model)
            return base + i * bar_width
        else:
            i = chinese_models.index(model)
            return base + len(western_models) * bar_width + intra_group_gap + i * bar_width

    western_block_width = len(western_models) * bar_width
    chinese_block_width = len(chinese_models) * bar_width

    def western_block_range(race_idx):
        return x[race_idx], x[race_idx] + western_block_width

    def chinese_block_range(race_idx):
        start = x[race_idx] + western_block_width + intra_group_gap
        return start, start + chinese_block_width

    # --------------------------------
    # 7. Plot
    # --------------------------------
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 背景色区分 Western / Chinese 区块
    for i in range(n_race):
        western_start = x[i] - bar_width * 0.5
        western_end   = x[i] + western_block_width
        chinese_start = western_end + intra_group_gap * 0.3
        chinese_end   = x[i] + block_width - bar_width * 0.5
        ax.axvspan(western_start, western_end, alpha=0.04, color="#4C8BE8", zorder=0)
        ax.axvspan(chinese_start, chinese_end, alpha=0.04, color="#D94801", zorder=0)

    # Bars
    for model in model_names:
        positions = [get_x_pos(i, model) for i in range(n_race)]
        ax.bar(
            positions,
            proportions[model],
            width=bar_width,
            color=model_colors[model],
            label=model,
            align="center"
        )

    # Real-world dashed baselines
    for i, race in enumerate(race_groups):

        # US + Europe
        y = real_race_us_eu[race]
        xmin, xmax = western_block_range(i)
        ax.hlines(
            y=y,
            xmin=xmin - bar_width / 2,
            xmax=xmax - bar_width / 2,
            linestyle="--",
            color="#4C72B0",
            linewidth=2.8,
            label="Real-world (US+Europe)" if i == 0 else None
        )

        # China
        y = real_race_china[race]
        xmin, xmax = chinese_block_range(i)
        ax.hlines(
            y=y,
            xmin=xmin - bar_width / 2,
            xmax=xmax - bar_width / 2,
            linestyle="--",
            color="#C44E52",
            linewidth=2.8,
            label="Real-world (China)" if i == 0 else None
        )

    # --------------------------------
    # 8. Axes
    # --------------------------------
    ax.set_xticks(x + block_width / 2)
    ax.set_xticklabels(race_groups, fontsize=27)
    ax.set_ylabel("Proportion", fontsize=23)
    ax.tick_params(axis="both", labelsize=27)

    # --------------------------------
    # 9. Legends（与 gender 图完全一致）
    # --------------------------------
    handles, labels = ax.get_legend_handles_labels()

    def H(lbl):
        return handles[labels.index(lbl)]

    # # Left legend: Western
    # leg_left = ax.legend(
    #     [
    #         H("flux"), H("Proteus"), H("sd3"), H("sana"),
    #         H("Real-world (US+Europe)")
    #     ],
    #     [
    #         legend_name_map["flux"], legend_name_map["Proteus"],
    #         legend_name_map["sd3"], legend_name_map["sana"],
    #         "Real-world (US+Europe)"
    #     ],
    #     loc="upper center",
    #     bbox_to_anchor=(0.41, 1),
    #     fontsize=15,
    #     framealpha=0.9,
    #     handlelength=1.2,
    #     handleheight=1.0
    # )
    # ax.add_artist(leg_left)
    #
    # # Right legend: Chinese
    # ax.legend(
    #     [
    #         H("hunyuan"), H("qwen"), H("kolors"), H("Wan2.1"),
    #         H("Real-world (China)")
    #     ],
    #     [
    #         legend_name_map["hunyuan"], legend_name_map["qwen"],
    #         legend_name_map["kolors"], legend_name_map["Wan2.1"],
    #         "Real-world (China)"
    #     ],
    #     loc="upper center",
    #     bbox_to_anchor=(0.380, 0.68),
    #     fontsize=15,
    #     framealpha=0.9,
    #     handlelength=1.2,
    #     handleheight=1.0
    # )

    plt.tight_layout()
    plt.savefig("neutral_race_distribution_final.png", dpi=300, bbox_inches="tight")
    plt.show()


def race_distribution():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # --------------------------------
    # 1. Models & groups
    # --------------------------------
    western_models = ["flux", "Proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "Wan2.1"]
    model_names = western_models + chinese_models

    legend_name_map = {
        "flux": "Flux",
        "Proteus": "Proteus",
        "sd3": "SD3",
        "sana": "SANA",
        "hunyuan": "Hunyuan",
        "qwen": "Qwen",
        "kolors": "Kolors",
        "Wan2.1": "Wan2.1",
    }

    # --------------------------------
    # 2. Colors
    # --------------------------------
    model_colors = {
        "flux": "#08306B",
        "Proteus": "#2171B5",
        "sd3": "#C6DBEF",
        "sana": "#6BAED6",
        "hunyuan": "#7F2704",
        "kolors": "#A63603",
        "qwen": "#E6550D",
        "Wan2.1": "#FDBE85",
    }

    # --------------------------------
    # 3. Load JSON
    # --------------------------------
    base_path = "/Users/wmt/projects/bias/data/jsons/race"
    data = {}

    for model in model_names:
        with open(os.path.join(base_path, f"{model}_emotion_race_counts.json"), "r") as f:
            data[model] = json.load(f)["person"]

    race_groups = list(data[model_names[0]].keys())
    n_race = len(race_groups)

    # --------------------------------
    # 4. Convert to proportions
    # --------------------------------
    proportions = {}
    for model, counts in data.items():
        total = sum(counts.values())
        proportions[model] = [counts[r] / total for r in race_groups]

    # --------------------------------
    # 5. Real-world race distributions
    # --------------------------------
    real_race_us_eu = {
        "White": 0.76,
        "Black": 0.08,
        "Asian": 0.06,
        "Others": 0.10
    }

    real_race_china = {
        "White": 0.01,
        "Black": 0.01,
        "Asian": 0.92,
        "Others": 0.06
    }

    # --------------------------------
    # 6. Layout parameters
    # --------------------------------
    bar_width = 0.09
    intra_group_gap = bar_width * 1.7
    inter_block_gap = bar_width * 5.0

    block_width = bar_width * 8 + intra_group_gap
    step = block_width + inter_block_gap
    x = np.arange(n_race) * step

    def get_x_pos(race_idx, model):
        base = x[race_idx]
        if model in western_models:
            i = western_models.index(model)
            return base + i * bar_width
        else:
            i = chinese_models.index(model)
            return base + len(western_models) * bar_width + intra_group_gap + i * bar_width

    western_block_width = len(western_models) * bar_width
    chinese_block_width = len(chinese_models) * bar_width

    def western_block_range(race_idx):
        return x[race_idx], x[race_idx] + western_block_width

    def chinese_block_range(race_idx):
        start = x[race_idx] + western_block_width + intra_group_gap
        return start, start + chinese_block_width

    # --------------------------------
    # 7. Plot
    # --------------------------------
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Bars
    for model in model_names:
        positions = [get_x_pos(i, model) for i in range(n_race)]
        ax.bar(
            positions,
            proportions[model],
            width=bar_width,
            color=model_colors[model],
            label=model,
            align="center"
        )

    # Real-world dashed baselines
    for i, race in enumerate(race_groups):

        # US + Europe
        y = real_race_us_eu[race]
        xmin, xmax = western_block_range(i)
        ax.hlines(
            y=y,
            xmin=xmin - bar_width / 2,
            xmax=xmax - bar_width / 2,
            linestyle="--",
            color="#4C72B0",
            linewidth=2.8,
            label="Real-world (US+Europe)" if i == 0 else None
        )

        # China
        y = real_race_china[race]
        xmin, xmax = chinese_block_range(i)
        ax.hlines(
            y=y,
            xmin=xmin - bar_width / 2,
            xmax=xmax - bar_width / 2,
            linestyle="--",
            color="#C44E52",
            linewidth=2.8,
            label="Real-world (China)" if i == 0 else None
        )

    # ↑ / ↓ arrows
    for model in model_names:
        for i, race in enumerate(race_groups):
            val = proportions[model][i]
            rv = real_race_us_eu[race] if model in western_models else real_race_china[race]
            xpos = get_x_pos(i, model)
            ax.annotate(
                "↑" if val > rv else "↓",
                (xpos, val + 0.015),
                ha="center",
                fontsize=17,
                color="#1e8449" if val > rv else "#6D2E46"
            )

    # --------------------------------
    # 8. Axes（统一 17 号）
    # --------------------------------
    ax.set_xticks(x + block_width / 2)
    ax.set_xticklabels(race_groups, fontsize=27)
    ax.set_ylabel("Proportion", fontsize=23)
    ax.tick_params(axis="both", labelsize=27)

    # --------------------------------
    # 9. Legends（与 gender 图完全一致）
    # --------------------------------
    handles, labels = ax.get_legend_handles_labels()

    def H(lbl):
        return handles[labels.index(lbl)]

    # Left legend: Western
    leg_left = ax.legend(
        [
            H("flux"), H("Proteus"), H("sd3"), H("sana"),
            H("Real-world (US+Europe)")
        ],
        [
            legend_name_map["flux"], legend_name_map["Proteus"],
            legend_name_map["sd3"], legend_name_map["sana"],
            "Real-world (US+Europe)"
        ],
        loc="upper center",
        bbox_to_anchor=(0.41, 1),
        fontsize=15,
        framealpha=0.9,
        handlelength=1.2,
        handleheight=1.0
    )
    ax.add_artist(leg_left)

    # Right legend: Chinese
    ax.legend(
        [
            H("hunyuan"), H("qwen"), H("kolors"), H("Wan2.1"),
            H("Real-world (China)")
        ],
        [
            legend_name_map["hunyuan"], legend_name_map["qwen"],
            legend_name_map["kolors"], legend_name_map["Wan2.1"],
            "Real-world (China)"
        ],
        loc="upper center",
        bbox_to_anchor=(0.380, 0.68),
        fontsize=15,
        framealpha=0.9,
        handlelength=1.2,
        handleheight=1.0
    )

    plt.tight_layout()
    plt.savefig("neutral_race_distribution_final.png", dpi=300, bbox_inches="tight")
    plt.show()



def age_distribution2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # --------------------------------
    # 1. Models & groups
    # --------------------------------
    western_models = ["flux", "Proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "Wan2.1"]
    model_names = western_models + chinese_models

    legend_name_map = {
        "flux":    "Flux",
        "Proteus": "Proteus",
        "sd3":     "SD3",
        "sana":    "SANA",
        "hunyuan": "Hunyuan",
        "qwen":    "Qwen",
        "kolors":  "Kolors",
        "Wan2.1":  "Wan2.1",
    }

    # --------------------------------
    # 2. Colors（与 gender / race 图一致）
    # --------------------------------
    model_colors = {
        "flux":    "#08306B",
        "Proteus": "#2171B5",
        "sd3":     "#6BAED6",
        "sana":    "#BDD7E7",
        "hunyuan": "#7F2704",
        "qwen":    "#D94801",
        "kolors":  "#FD8D3C",
        "Wan2.1":  "#FDD0A2",
    }

    # --------------------------------
    # 3. Load data
    # --------------------------------
    base_path = "/Users/wmt/projects/bias/data/jsons/age"
    data = {}
    for model in model_names:
        with open(os.path.join(base_path, f"{model}_age_counts.json"), "r") as f:
            data[model] = json.load(f)["person"]

    age_groups = list(data[model_names[0]].keys())
    n_age = len(age_groups)

    # --------------------------------
    # 4. Convert to proportions
    # --------------------------------
    proportions = {}
    for model, counts in data.items():
        total = sum(counts.values())
        proportions[model] = [counts[g] / total for g in age_groups]

    # --------------------------------
    # 5. Real-world age distributions
    # --------------------------------
    real_age_us_eu = {
        "0-9": 0.12, "10-19": 0.12, "20-39": 0.27, "40-59": 0.26, "60+": 0.23
    }

    real_age_china = {
        "0-9": 0.11, "10-19": 0.12, "20-39": 0.31, "40-59": 0.28, "60+": 0.18
    }

    # --------------------------------
    # 6. Layout parameters
    # --------------------------------
    bar_width = 0.09
    intra_group_gap = bar_width * 1.7
    inter_block_gap = bar_width * 5.0

    block_width = bar_width * 8 + intra_group_gap
    step = block_width + inter_block_gap
    x = np.arange(n_age) * step

    def get_x_pos(age_idx, model):
        base = x[age_idx]
        if model in western_models:
            i = western_models.index(model)
            return base + i * bar_width
        else:
            i = chinese_models.index(model)
            return base + len(western_models) * bar_width + intra_group_gap + i * bar_width

    western_block_width = len(western_models) * bar_width
    chinese_block_width = len(chinese_models) * bar_width

    def western_block_range(age_idx):
        return x[age_idx], x[age_idx] + western_block_width

    def chinese_block_range(age_idx):
        start = x[age_idx] + western_block_width + intra_group_gap
        return start, start + chinese_block_width

    # --------------------------------
    # 7. Plot
    # --------------------------------
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 背景色区分 Western / Chinese 区块
    for i in range(n_age):
        western_start = x[i] - bar_width * 0.5
        western_end   = x[i] + western_block_width
        chinese_start = western_end + intra_group_gap * 0.3
        chinese_end   = x[i] + block_width - bar_width * 0.5
        ax.axvspan(western_start, western_end, alpha=0.04, color="#4C8BE8", zorder=0)
        ax.axvspan(chinese_start, chinese_end, alpha=0.04, color="#D94801", zorder=0)

    # Bars
    for model in model_names:
        positions = [get_x_pos(i, model) for i in range(n_age)]
        ax.bar(
            positions,
            proportions[model],
            width=bar_width,
            color=model_colors[model],
            label=model,
            align="center"
        )

    # Real-world dashed baselines
    for i, age in enumerate(age_groups):

        # US + Europe
        y = real_age_us_eu[age]
        xmin, xmax = western_block_range(i)
        ax.hlines(
            y=y,
            xmin=xmin - bar_width / 2,
            xmax=xmax - bar_width / 2,
            linestyle="--",
            color="#4C72B0",
            linewidth=2.8,
            label="Real-world (US+Europe)" if i == 0 else None
        )

        # China
        y = real_age_china[age]
        xmin, xmax = chinese_block_range(i)
        ax.hlines(
            y=y,
            xmin=xmin - bar_width / 2,
            xmax=xmax - bar_width / 2,
            linestyle="--",
            color="#C44E52",
            linewidth=2.8,
            label="Real-world (China)" if i == 0 else None
        )

    # --------------------------------
    # 8. Axes
    # --------------------------------
    ax.set_xticks(x + block_width / 2)
    ax.set_xticklabels(age_groups, fontsize=27)
    ax.set_ylabel("Proportion", fontsize=23)
    ax.tick_params(axis="both", labelsize=27)

    # --------------------------------
    # 9. Legends（与 gender / race 完全一致）
    # --------------------------------
    handles, labels = ax.get_legend_handles_labels()

    def H(lbl):
        return handles[labels.index(lbl)]

    # Left legend: Western
    leg_left = ax.legend(
        [
            H("flux"), H("Proteus"), H("sd3"), H("sana"),
            H("Real-world (US+Europe)")
        ],
        [
            legend_name_map["flux"], legend_name_map["Proteus"],
            legend_name_map["sd3"], legend_name_map["sana"],
            "Real-world (US+Europe)"
        ],
        loc="upper center",
        bbox_to_anchor=(0.8, 1),
        fontsize=15,
        framealpha=0.9,
        handlelength=1.2,
        handleheight=1.0
    )
    ax.add_artist(leg_left)

    # Right legend: Chinese
    ax.legend(
        [
            H("hunyuan"), H("qwen"), H("kolors"), H("Wan2.1"),
            H("Real-world (China)")
        ],
        [
            legend_name_map["hunyuan"], legend_name_map["qwen"],
            legend_name_map["kolors"], legend_name_map["Wan2.1"],
            "Real-world (China)"
        ],
        loc="upper center",
        bbox_to_anchor=(0.77, 0.68),
        fontsize=15,
        framealpha=0.9,
        handlelength=1.2,
        handleheight=1.0
    )

    plt.tight_layout()
    plt.savefig("neutral_age_distribution_final.png", dpi=300, bbox_inches="tight")
    plt.show()


def age_distribution():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # --------------------------------
    # 1. Models & groups
    # --------------------------------
    western_models = ["flux", "Proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "Wan2.1"]
    model_names = western_models + chinese_models

    legend_name_map = {
        "flux": "Flux",
        "Proteus": "Proteus",
        "sd3": "SD3",
        "sana": "SANA",
        "hunyuan": "Hunyuan",
        "qwen": "Qwen",
        "kolors": "Kolors",
        "Wan2.1": "Wan2.1",
    }

    # --------------------------------
    # 2. Colors
    # --------------------------------
    model_colors = {
        "flux": "#08306B",
        "Proteus": "#2171B5",
        "sd3": "#C6DBEF",
        "sana": "#6BAED6",
        "hunyuan": "#7F2704",
        "kolors": "#A63603",
        "qwen": "#E6550D",
        "Wan2.1": "#FDBE85",
    }

    # --------------------------------
    # 3. Load data
    # --------------------------------
    base_path = "/Users/wmt/projects/bias/data/jsons/age"
    data = {}
    for model in model_names:
        with open(os.path.join(base_path, f"{model}_age_counts.json"), "r") as f:
            data[model] = json.load(f)["person"]

    age_groups = list(data[model_names[0]].keys())
    n_age = len(age_groups)

    # --------------------------------
    # 4. Convert to proportions
    # --------------------------------
    proportions = {}
    for model, counts in data.items():
        total = sum(counts.values())
        proportions[model] = [counts[g] / total for g in age_groups]

    # --------------------------------
    # 5. Real-world age distributions
    # --------------------------------
    real_age_us_eu = {
        "0-9": 0.12, "10-19": 0.12, "20-39": 0.27, "40-59": 0.26, "60+": 0.23
    }

    real_age_china = {
        "0-9": 0.11, "10-19": 0.12, "20-39": 0.31, "40-59": 0.28, "60+": 0.18
    }

    # --------------------------------
    # 6. Layout parameters
    # --------------------------------
    bar_width = 0.09
    intra_group_gap = bar_width * 1.7
    inter_block_gap = bar_width * 5.0

    block_width = bar_width * 8 + intra_group_gap
    step = block_width + inter_block_gap
    x = np.arange(n_age) * step

    def get_x_pos(age_idx, model):
        base = x[age_idx]
        if model in western_models:
            i = western_models.index(model)
            return base + i * bar_width
        else:
            i = chinese_models.index(model)
            return base + len(western_models) * bar_width + intra_group_gap + i * bar_width

    western_block_width = len(western_models) * bar_width
    chinese_block_width = len(chinese_models) * bar_width

    def western_block_range(age_idx):
        return x[age_idx], x[age_idx] + western_block_width

    def chinese_block_range(age_idx):
        start = x[age_idx] + western_block_width + intra_group_gap
        return start, start + chinese_block_width

    # --------------------------------
    # 7. Plot
    # --------------------------------
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Bars
    for model in model_names:
        positions = [get_x_pos(i, model) for i in range(n_age)]
        ax.bar(
            positions,
            proportions[model],
            width=bar_width,
            color=model_colors[model],
            label=model,
            align="center"
        )

    # Real-world dashed baselines
    for i, age in enumerate(age_groups):

        # US + Europe
        y = real_age_us_eu[age]
        xmin, xmax = western_block_range(i)
        ax.hlines(
            y=y,
            xmin=xmin - bar_width / 2,
            xmax=xmax - bar_width / 2,
            linestyle="--",
            color="#4C72B0",
            linewidth=2.8,
            label="Real-world (US+Europe)" if i == 0 else None
        )

        # China
        y = real_age_china[age]
        xmin, xmax = chinese_block_range(i)
        ax.hlines(
            y=y,
            xmin=xmin - bar_width / 2,
            xmax=xmax - bar_width / 2,
            linestyle="--",
            color="#C44E52",
            linewidth=2.8,
            label="Real-world (China)" if i == 0 else None
        )

    # ↑ / ↓ arrows
    for model in model_names:
        for i, age in enumerate(age_groups):
            val = proportions[model][i]
            rv = real_age_us_eu[age] if model in western_models else real_age_china[age]
            xpos = get_x_pos(i, model)
            ax.annotate(
                "↑" if val > rv else "↓",
                (xpos, val + 0.015),
                ha="center",
                fontsize=17,
                color="#1e8449" if val > rv else "#6D2E46"
            )

    # --------------------------------
    # 8. Axes（统一 17 号）
    # --------------------------------
    ax.set_xticks(x + block_width / 2)
    ax.set_xticklabels(age_groups, fontsize=27)
    ax.set_ylabel("Proportion", fontsize=23)
    ax.tick_params(axis="both", labelsize=27)

    # --------------------------------
    # 9. Legends（与 gender / race 完全一致）
    # --------------------------------
    handles, labels = ax.get_legend_handles_labels()

    def H(lbl):
        return handles[labels.index(lbl)]

    # Left legend: Western
    leg_left = ax.legend(
        [
            H("flux"), H("Proteus"), H("sd3"), H("sana"),
            H("Real-world (US+Europe)")
        ],
        [
            legend_name_map["flux"], legend_name_map["Proteus"],
            legend_name_map["sd3"], legend_name_map["sana"],
            "Real-world (US+Europe)"
        ],
        loc="upper center",
        bbox_to_anchor=(0.8, 1),
        fontsize=15,
        framealpha=0.9,
        handlelength=1.2,
        handleheight=1.0
    )
    ax.add_artist(leg_left)

    # Right legend: Chinese
    ax.legend(
        [
            H("hunyuan"), H("qwen"), H("kolors"), H("Wan2.1"),
            H("Real-world (China)")
        ],
        [
            legend_name_map["hunyuan"], legend_name_map["qwen"],
            legend_name_map["kolors"], legend_name_map["Wan2.1"],
            "Real-world (China)"
        ],
        loc="upper center",
        bbox_to_anchor=(0.77, 0.68),
        fontsize=15,
        framealpha=0.9,
        handlelength=1.2,
        handleheight=1.0
    )

    plt.tight_layout()
    plt.savefig("neutral_age_distribution_final.png", dpi=300, bbox_inches="tight")
    plt.show()



if __name__ == '__main__':
    # gender_distribution()
    gender_distribution2()
    # race_distribution2()
    # race_distribution()
    # age_distribution()
    age_distribution2()