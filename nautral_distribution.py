import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd



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
    western_models = ["flux", "proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]
    model_names = western_models + chinese_models

    # --------------------------------
    # 2. Colors
    # --------------------------------
    model_colors = {
        "flux": "#08306B",
        "proteus": "#2171B5",
        "sana": "#6BAED6",
        "sd3": "#C6DBEF",
        "hunyuan": "#7F2704",
        "kolors": "#A63603",
        "qwen": "#E6550D",
        "wan2.1": "#FDBE85",
    }

    # --------------------------------
    # 3. Load data
    # --------------------------------
    base_path = "/Users/wmt/projects/bias/write_into_json/age"
    data = {}
    emotion = "neutral"
    for model in model_names:
        file_path = os.path.join(base_path, f"{model}_age.json")
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        if emotion not in json_data:
            raise ValueError(f"{emotion} not found in {file_path}")

        data[model] = json_data[emotion]

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
    # 5. Real-world distribution
    # --------------------------------
    real_world_age = {
        "0-9":   0.165959,
        "10-19": 0.163733,
        "20-39": 0.297204,
        "40-59": 0.231504,
        "60+":   0.141601
    }
    real_values = [real_world_age[g] for g in age_groups]

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

    # --------------------------------
    # 7. Plot
    # --------------------------------
    plt.figure(figsize=(10, 6))  # 👈 稍微放大整体画布

    legend_name_map = {
        "flux": "Flux",
        "proteus": "Proteus",
        "sd3": "SD3",
        "sana": "SANA",
        "hunyuan": "Hunyuan",
        "qwen": "Qwen",
        "kolors": "Kolors",
        "wan2.1": "Wan2.1",
    }

    # Bars
    for model in model_names:
        positions = [get_x_pos(i, model) for i in range(n_age)]
        plt.bar(
            positions,
            proportions[model],
            width=bar_width,
            color=model_colors[model],
            label=model
        )

    # Real-world baselines
    for i, rv in enumerate(real_values):
        plt.hlines(
            y=rv,
            xmin=x[i] - bar_width * 0.5,
            xmax=x[i] + block_width - bar_width * 0.5,
            linestyle="--",
            color="#55A868",
            linewidth=2.5,
            alpha=0.9
        )

    # ↑ / ↓ arrows
    for model in model_names:
        for i, rv in enumerate(real_values):
            val = proportions[model][i]
            xpos = get_x_pos(i, model)
            plt.annotate(
                "↑" if val > rv else "↓",
                (xpos, val + 0.015),
                ha="center",
                fontsize=14,
                color="#1e8449" if val > rv else "#6D2E46"
            )

    # --------------------------------
    # 8. Axis fonts (论文主图标准字号)
    # --------------------------------
    plt.xticks(x + block_width / 2, age_groups, fontsize=30)
    plt.ylabel("Proportion", fontsize=28)
    plt.yticks(fontsize=26)

    # --------------------------------
    # 9. Legend (字体 + 图标一起变大)
    # --------------------------------
    handles, labels = plt.gca().get_legend_handles_labels()
    ordered_handles = [handles[labels.index(m)] for m in model_names]
    ordered_labels = [legend_name_map[m] for m in model_names]

    plt.legend(
        ordered_handles,
        ordered_labels,
        ncol=2,
        fontsize=19,          # 👈 legend 字体
        loc="upper center",
        bbox_to_anchor=(0.85, 1),
        framealpha=0.9,
        handlelength=1.8,     # 👈 色块宽度
        handleheight=1.0,     # 👈 色块高度
        borderpad=0.6
    )

    plt.tight_layout()
    plt.savefig("neutral_age_distribution_final.png", dpi=300, bbox_inches="tight")
    plt.show()

def age_distribution2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from matplotlib.lines import Line2D

    sns.set_style("whitegrid")

    # --------------------------------
    # 1. Models & groups
    # --------------------------------
    western_models = ["flux", "proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]
    model_names = western_models + chinese_models

    # --------------------------------
    # 2. Colors
    # --------------------------------
    model_colors = {
        "flux":    "#08306B",
        "proteus": "#2171B5",
        "sd3":     "#6BAED6",
        "sana":    "#BDD7E7",
        "hunyuan": "#7F2704",
        "qwen":    "#D94801",
        "kolors":  "#FD8D3C",
        "wan2.1":  "#FDD0A2",
    }

    legend_name_map = {
        "flux": "Flux", "proteus": "Proteus",
        "sd3": "SD3",   "sana": "SANA",
        "hunyuan": "Hunyuan", "qwen": "Qwen",
        "kolors": "Kolors",   "wan2.1": "Wan2.1",
    }

    # --------------------------------
    # 3. Load data
    # --------------------------------
    base_path = "/Users/wmt/projects/bias/write_into_json/age"
    data = {}
    emotion = "neutral"
    for model in model_names:
        file_path = os.path.join(base_path, f"{model}_age.json")
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        data[model] = json_data[emotion]

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
    # 5. Real-world distribution
    # --------------------------------
    real_world_age = {
        "0-9":   0.165959,
        "10-19": 0.163733,
        "20-39": 0.297204,
        "40-59": 0.231504,
        "60+":   0.141601
    }
    real_values = [real_world_age[g] for g in age_groups]

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

    # --------------------------------
    # 7. Plot
    # --------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # 背景色区分 Western / Chinese
    for i in range(n_age):
        western_start = x[i] - bar_width * 0.5
        western_end   = x[i] + len(western_models) * bar_width
        chinese_start = western_end + intra_group_gap * 0.3
        chinese_end   = x[i] + block_width - bar_width * 0.5
        ax.axvspan(western_start, western_end, alpha=0.04, color="#4C8BE8", zorder=0)
        ax.axvspan(chinese_start, chinese_end, alpha=0.04, color="#D94801", zorder=0)

    # Bars - 直接用 model_names 顺序，label 用 legend_name_map
    for model in model_names:
        positions = [get_x_pos(i, model) for i in range(n_age)]
        ax.bar(
            positions,
            proportions[model],
            width=bar_width,
            color=model_colors[model],
            label=legend_name_map[model],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

    # Real-world baselines
    for i, rv in enumerate(real_values):
        ax.hlines(
            y=rv,
            xmin=x[i] - bar_width * 0.5,
            xmax=x[i] + block_width - bar_width * 0.5,
            linestyle="--",
            color="#2CA02C",
            linewidth=2.0,
            alpha=0.85,
            zorder=4,
            label="World population" if i == 0 else "_nolegend_",
        )

    # --------------------------------
    # 8. Axis
    # --------------------------------
    ax.set_xticks(x + block_width / 2)
    ax.set_xticklabels(age_groups, fontsize=30)
    ax.set_ylabel("Proportion", fontsize=28)
    ax.tick_params(axis="y", labelsize=26)
    ax.set_xlim(x[0] - bar_width * 2, x[-1] + block_width + bar_width)
    ax.set_ylim(0, 1.05)
    # ax.set_title("(c) Age", fontsize=22, fontweight="bold", loc="left", pad=10)

    # --------------------------------
    # 9. Legend
    # --------------------------------
    from matplotlib.lines import Line2D

    def make_blank():
        return Line2D([0], [0], color="none", linewidth=0)

    # 先按 model_names 顺序拿到所有handle
    all_handles, all_labels = plt.gca().get_legend_handles_labels()
    handle_dict = dict(zip(all_labels, all_handles))

    w_handles = [handle_dict[legend_name_map[m]] for m in western_models]
    w_labels = [legend_name_map[m] for m in western_models]
    c_handles = [handle_dict[legend_name_map[m]] for m in chinese_models]
    c_labels = [legend_name_map[m] for m in chinese_models]
    world_handle = handle_dict["World population"]

    western_patch = mpatches.Patch(color="#4C8BE8", alpha=0.3)
    chinese_patch = mpatches.Patch(color="#D94801", alpha=0.3)

    final_handles = (
        # 左列6个
            [western_patch, w_handles[0], w_handles[1], w_handles[2], w_handles[3], world_handle] +
            # 右列6个
            [chinese_patch, c_handles[0], c_handles[1], c_handles[2], c_handles[3], make_blank()]
    )
    final_labels = (
            ["Western", w_labels[0], w_labels[1], w_labels[2], w_labels[3], "World population"] +
            ["Chinese", c_labels[0], c_labels[1], c_labels[2], c_labels[3], ""]
    )

    leg = plt.legend(  # 用 plt.legend 而不是 ax.legend
        final_handles,
        final_labels,
        ncol=2,
        fontsize=14,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=0.9,
        handlelength=1.5,
        handleheight=1.0,
        borderpad=0.7,
        labelspacing=0.4,
        columnspacing=1.0,
    )

    for text in leg.get_texts():
        if text.get_text() in ("Western", "Chinese"):
            text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig("neutral_age_distribution_improved.png", dpi=300, bbox_inches="tight")
    plt.show()




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
    western_models = ["flux", "proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]
    model_names = western_models + chinese_models

    legend_name_map = {
        "flux": "Flux",
        "proteus": "Proteus",
        "sd3": "SD3",
        "sana": "SANA",
        "hunyuan": "Hunyuan",
        "qwen": "Qwen",
        "kolors": "Kolors",
        "wan2.1": "Wan2.1",
    }

    # --------------------------------
    # 2. Colors
    # --------------------------------
    model_colors = {
        "flux": "#08306B",
        "proteus": "#2171B5",
        "sd3": "#C6DBEF",
        "sana": "#6BAED6",
        "hunyuan": "#7F2704",
        "kolors": "#A63603",
        "qwen": "#E6550D",
        "wan2.1": "#FDBE85",
    }

    # --------------------------------
    # 3. Load data
    # --------------------------------
    base_path = "/Users/wmt/projects/bias/write_into_json/gender"
    data = {}
    emotion = "neutral"
    for model in model_names:
        file_path = os.path.join(base_path, f"{model}_gender.json")
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        if emotion not in json_data:
            raise ValueError(f"{emotion} not found in {file_path}")

        data[model] = json_data[emotion]

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
    # 5. Real-world gender distribution
    # --------------------------------
    real_world_gender = {
        "Male": 0.5029,
        "Female": 0.4971
    }
    real_values = [real_world_gender[g] for g in gender_groups]

    # --------------------------------
    # 6. Layout parameters (same as age)
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
            return (
                base
                + len(western_models) * bar_width
                + intra_group_gap
                + i * bar_width
            )

    # --------------------------------
    # 7. Plot
    # --------------------------------
    plt.figure(figsize=(10, 6))

    # Bars
    for model in model_names:
        positions = [get_x_pos(i, model) for i in range(n_gender)]
        plt.bar(
            positions,
            proportions[model],
            width=bar_width,
            color=model_colors[model],
            label=model
        )

    # Real-world baselines
    for i, rv in enumerate(real_values):
        plt.hlines(
            y=rv,
            xmin=x[i] - bar_width * 0.5,
            xmax=x[i] + block_width - bar_width * 0.5,
            linestyle="--",
            color="#55A868",
            linewidth=2.5,
            alpha=0.9
        )

    # ↑ / ↓ arrows
    for model in model_names:
        for i, rv in enumerate(real_values):
            val = proportions[model][i]
            xpos = get_x_pos(i, model)
            plt.annotate(
                "↑" if val > rv else "↓",
                (xpos, val + 0.015),
                ha="center",
                fontsize=14,
                color="#1e8449" if val > rv else "#6D2E46"
            )

    # --------------------------------
    # 8. Axis fonts (与 age 对齐)
    # --------------------------------
    plt.xticks(x + block_width / 2, gender_groups, fontsize=30)
    plt.ylabel("Proportion", fontsize=28)
    plt.yticks(fontsize=26)

    # --------------------------------
    # 9. Legend (与 age 完全一致)
    # --------------------------------
    handles, labels = plt.gca().get_legend_handles_labels()
    ordered_handles = [handles[labels.index(m)] for m in model_names]
    ordered_labels = [legend_name_map[m] for m in model_names]

    # plt.legend(
    #     ordered_handles,
    #     ordered_labels,
    #     ncol=2,
    #     fontsize=14,
    #     loc="upper center",
    #     bbox_to_anchor=(0.52, 1),
    #     framealpha=0.9,
    #     handlelength=1.8,
    #     handleheight=1.0,
    #     borderpad=0.6
    # )

    plt.tight_layout()
    plt.savefig("neutral_gender_distribution_final.png", dpi=300, bbox_inches="tight")
    plt.show()


def gender_distribution2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from matplotlib.lines import Line2D

    sns.set_style("whitegrid")

    # --------------------------------
    # 1. Models & groups
    # --------------------------------
    western_models = ["flux", "proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]
    model_names = western_models + chinese_models

    legend_name_map = {
        "flux": "Flux", "proteus": "Proteus",
        "sd3": "SD3",   "sana": "SANA",
        "hunyuan": "Hunyuan", "qwen": "Qwen",
        "kolors": "Kolors",   "wan2.1": "Wan2.1",
    }

    # --------------------------------
    # 2. Colors
    # --------------------------------
    model_colors = {
        "flux":    "#08306B",
        "proteus": "#2171B5",
        "sd3":     "#6BAED6",
        "sana":    "#BDD7E7",
        "hunyuan": "#7F2704",
        "qwen":    "#D94801",
        "kolors":  "#FD8D3C",
        "wan2.1":  "#FDD0A2",
    }

    # --------------------------------
    # 3. Load data
    # --------------------------------
    base_path = "/Users/wmt/projects/bias/write_into_json/gender"
    data = {}
    emotion = "neutral"
    for model in model_names:
        file_path = os.path.join(base_path, f"{model}_gender.json")
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        if emotion not in json_data:
            raise ValueError(f"{emotion} not found in {file_path}")
        data[model] = json_data[emotion]

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
    # 5. Real-world distribution
    # --------------------------------
    real_world_gender = {
        "Male":   0.5029,
        "Female": 0.4971
    }
    real_values = [real_world_gender[g] for g in gender_groups]

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

    # --------------------------------
    # 7. Plot
    # --------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # 背景色区分 Western / Chinese
    for i in range(n_gender):
        western_start = x[i] - bar_width * 0.5
        western_end   = x[i] + len(western_models) * bar_width
        chinese_start = western_end + intra_group_gap * 0.3
        chinese_end   = x[i] + block_width - bar_width * 0.5
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
            label=legend_name_map[model],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

    # Real-world baselines
    for i, rv in enumerate(real_values):
        ax.hlines(
            y=rv,
            xmin=x[i] - bar_width * 0.5,
            xmax=x[i] + block_width - bar_width * 0.5,
            linestyle="--",
            color="#2CA02C",
            linewidth=2.0,
            alpha=0.85,
            zorder=4,
            label="World population" if i == 0 else "_nolegend_",
        )

    # --------------------------------
    # 8. Axis
    # --------------------------------
    ax.set_xticks(x + block_width / 2)
    ax.set_xticklabels(gender_groups, fontsize=30)
    ax.set_ylabel("Proportion", fontsize=28)
    ax.tick_params(axis="y", labelsize=26)
    ax.set_xlim(x[0] - bar_width * 2, x[-1] + block_width + bar_width)
    ax.set_ylim(0, 1.05)
    # ax.set_title("(a) Gender", fontsize=22, fontweight="bold", loc="left", pad=10)

    # --------------------------------
    # 9. Legend
    # --------------------------------
    def make_blank():
        return Line2D([0], [0], color="none", linewidth=0)

    all_handles, all_labels = plt.gca().get_legend_handles_labels()
    valid_names = [legend_name_map[m] for m in model_names] + ["World population"]
    filtered = [(h, l) for h, l in zip(all_handles, all_labels) if l in valid_names]
    all_handles, all_labels = zip(*filtered)
    handle_dict = dict(zip(all_labels, all_handles))

    world_handle = handle_dict["World population"]
    w_handles = [handle_dict[legend_name_map[m]] for m in western_models]
    w_labels  = [legend_name_map[m] for m in western_models]
    c_handles = [handle_dict[legend_name_map[m]] for m in chinese_models]
    c_labels  = [legend_name_map[m] for m in chinese_models]

    western_patch = mpatches.Patch(color="#4C8BE8", alpha=0.3)
    chinese_patch = mpatches.Patch(color="#D94801", alpha=0.3)

    final_handles = (
        [western_patch, w_handles[0], w_handles[1], w_handles[2], w_handles[3], world_handle] +
        [chinese_patch, c_handles[0], c_handles[1], c_handles[2], c_handles[3], make_blank()]
    )
    final_labels = (
        ["Western",  w_labels[0], w_labels[1], w_labels[2], w_labels[3], "World population"] +
        ["Chinese",  c_labels[0], c_labels[1], c_labels[2], c_labels[3], ""]
    )

    # leg = plt.legend(
    #     final_handles,
    #     final_labels,
    #     ncol=2,
    #     fontsize=14,
    #     loc="upper right",
    #     bbox_to_anchor=(1.0, 1.0),
    #     framealpha=0.9,
    #     handlelength=1.5,
    #     handleheight=1.0,
    #     borderpad=0.7,
    #     labelspacing=0.4,
    #     columnspacing=1.0,
    # )

    # for text in leg.get_texts():
    #     if text.get_text() in ("Western", "Chinese"):
    #         text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig("neutral_gender_distribution_improved.png", dpi=300, bbox_inches="tight")
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
    western_models = ["flux", "proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]
    model_names = western_models + chinese_models

    legend_name_map = {
        "flux": "Flux",
        "proteus": "Proteus",
        "sd3": "SD3",
        "sana": "SANA",
        "hunyuan": "Hunyuan",
        "qwen": "Qwen",
        "kolors": "Kolors",
        "wan2.1": "Wan2.1",
    }

    # --------------------------------
    # 2. Colors
    # --------------------------------
    model_colors = {
        "flux": "#08306B",
        "proteus": "#2171B5",
        "sd3": "#C6DBEF",
        "sana": "#6BAED6",
        "hunyuan": "#7F2704",
        "kolors": "#A63603",
        "qwen": "#E6550D",
        "wan2.1": "#FDBE85",
    }

    # --------------------------------
    # 3. Load JSON
    # --------------------------------
    base_path = "/Users/wmt/projects/bias/write_into_json/race"
    data = {}

    emotion = "neutral"
    for model in model_names:
        file_path = os.path.join(base_path, f"{model}_race.json")
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        if emotion not in json_data:
            raise ValueError(f"{emotion} not found in {file_path}")

        data[model] = json_data[emotion]

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
    # 5. Real-world race distribution
    # --------------------------------
    real_world = {
        "White": 0.1307,
        "Black": 0.1689,
        "Asian": 0.3199,
        "Others": 0.3805
    }
    real_values = [real_world[r] for r in race_groups]

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

    # --------------------------------
    # 7. Plot
    # --------------------------------
    plt.figure(figsize=(10, 6))  # 👈 与 age 一致

    # Bars
    for model in model_names:
        positions = [get_x_pos(i, model) for i in range(n_race)]
        plt.bar(
            positions,
            proportions[model],
            width=bar_width,
            color=model_colors[model],
            label=model
        )

    # Real-world dashed baselines
    for i, rv in enumerate(real_values):
        plt.hlines(
            y=rv,
            xmin=x[i] - bar_width * 0.5,
            xmax=x[i] + block_width - bar_width * 0.5,
            linestyle="--",
            color="#55A868",
            linewidth=2.5,   # 👈 对齐 age
            alpha=0.9
        )

    # ↑ / ↓ arrows
    for model in model_names:
        for i, rv in enumerate(real_values):
            val = proportions[model][i]
            xpos = get_x_pos(i, model)
            plt.annotate(
                "↑" if val > rv else "↓",
                (xpos, val + 0.015),
                ha="center",
                fontsize=14,   # 👈 对齐 age
                color="#1e8449" if val > rv else "#6D2E46"
            )

    # --------------------------------
    # 8. Axis fonts（完全对齐 age）
    # --------------------------------
    plt.xticks(x + block_width / 2, race_groups, fontsize=30)
    plt.ylabel("Proportion", fontsize=28)
    plt.yticks(fontsize=26)

    # --------------------------------
    # 9. Legend（完全对齐 age）
    # --------------------------------
    handles, labels = plt.gca().get_legend_handles_labels()
    ordered_handles = [handles[labels.index(m)] for m in model_names]
    ordered_labels = [legend_name_map[m] for m in model_names]

    # plt.legend(
    #     ordered_handles,
    #     ordered_labels,
    #     ncol=2,
    #     fontsize=16,
    #     loc="upper center",
    #     bbox_to_anchor=(0.4, 1),
    #     framealpha=0.9,
    #     handlelength=1.8,
    #     handleheight=1.0,
    #     borderpad=0.6
    # )

    plt.tight_layout()
    plt.savefig("neutral_race_distribution_final.png", dpi=300, bbox_inches="tight")
    plt.show()


def race_distribution2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from matplotlib.lines import Line2D

    sns.set_style("whitegrid")

    # --------------------------------
    # 1. Models & groups
    # --------------------------------
    western_models = ["flux", "proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]
    model_names = western_models + chinese_models

    legend_name_map = {
        "flux": "Flux", "proteus": "Proteus",
        "sd3": "SD3",   "sana": "SANA",
        "hunyuan": "Hunyuan", "qwen": "Qwen",
        "kolors": "Kolors",   "wan2.1": "Wan2.1",
    }

    # --------------------------------
    # 2. Colors
    # --------------------------------
    model_colors = {
        "flux":    "#08306B",
        "proteus": "#2171B5",
        "sd3":     "#6BAED6",
        "sana":    "#BDD7E7",
        "hunyuan": "#7F2704",
        "qwen":    "#D94801",
        "kolors":  "#FD8D3C",
        "wan2.1":  "#FDD0A2",
    }

    # --------------------------------
    # 3. Load JSON
    # --------------------------------
    base_path = "/Users/wmt/projects/bias/write_into_json/race"
    data = {}
    emotion = "neutral"
    for model in model_names:
        file_path = os.path.join(base_path, f"{model}_race.json")
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        if emotion not in json_data:
            raise ValueError(f"{emotion} not found in {file_path}")
        data[model] = json_data[emotion]

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
    # 5. Real-world distribution
    # --------------------------------
    real_world = {
        "White":  0.1307,
        "Black":  0.1689,
        "Asian":  0.3199,
        "Others": 0.3805
    }
    real_values = [real_world[r] for r in race_groups]

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

    # --------------------------------
    # 7. Plot
    # --------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # 背景色区分 Western / Chinese
    for i in range(n_race):
        western_start = x[i] - bar_width * 0.5
        western_end   = x[i] + len(western_models) * bar_width
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
            label=legend_name_map[model],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

    # Real-world baselines
    for i, rv in enumerate(real_values):
        ax.hlines(
            y=rv,
            xmin=x[i] - bar_width * 0.5,
            xmax=x[i] + block_width - bar_width * 0.5,
            linestyle="--",
            color="#2CA02C",
            linewidth=2.0,
            alpha=0.85,
            zorder=4,
            label="World population" if i == 0 else "_nolegend_",
        )

    # --------------------------------
    # 8. Axis
    # --------------------------------
    ax.set_xticks(x + block_width / 2)
    ax.set_xticklabels(race_groups, fontsize=30)
    ax.set_ylabel("Proportion", fontsize=28)
    ax.tick_params(axis="y", labelsize=26)
    ax.set_xlim(x[0] - bar_width * 2, x[-1] + block_width + bar_width)
    ax.set_ylim(0, 1.05)
    # ax.set_title("(b) Race", fontsize=22, fontweight="bold", loc="left", pad=10)

    # --------------------------------
    # 9. Legend
    # --------------------------------
    def make_blank():
        return Line2D([0], [0], color="none", linewidth=0)

    all_handles, all_labels = plt.gca().get_legend_handles_labels()
    valid_names = [legend_name_map[m] for m in model_names] + ["World population"]
    filtered = [(h, l) for h, l in zip(all_handles, all_labels) if l in valid_names]
    all_handles, all_labels = zip(*filtered)
    handle_dict = dict(zip(all_labels, all_handles))

    world_handle = handle_dict["World population"]
    w_handles = [handle_dict[legend_name_map[m]] for m in western_models]
    w_labels  = [legend_name_map[m] for m in western_models]
    c_handles = [handle_dict[legend_name_map[m]] for m in chinese_models]
    c_labels  = [legend_name_map[m] for m in chinese_models]

    western_patch = mpatches.Patch(color="#4C8BE8", alpha=0.3)
    chinese_patch = mpatches.Patch(color="#D94801", alpha=0.3)

    final_handles = (
        [western_patch, w_handles[0], w_handles[1], w_handles[2], w_handles[3], world_handle] +
        [chinese_patch, c_handles[0], c_handles[1], c_handles[2], c_handles[3], make_blank()]
    )
    final_labels = (
        ["Western",  w_labels[0], w_labels[1], w_labels[2], w_labels[3], "World population"] +
        ["Chinese",  c_labels[0], c_labels[1], c_labels[2], c_labels[3], ""]
    )

    # leg = plt.legend(
    #     final_handles,
    #     final_labels,
    #     ncol=2,
    #     fontsize=14,
    #     loc="upper right",
    #     bbox_to_anchor=(1.05, 1.0),
    #     framealpha=0.9,
    #     handlelength=1.5,
    #     handleheight=1.0,
    #     borderpad=0.7,
    #     labelspacing=0.4,
    #     columnspacing=1.0,
    # )
    #
    # for text in leg.get_texts():
    #     if text.get_text() in ("Western", "Chinese"):
    #         text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig("neutral_race_distribution_improved.png", dpi=300, bbox_inches="tight")
    plt.show()



def attractive_distribution():
    sns.set_style("whitegrid")  # grid background (good for papers)

    # --------------------------------
    # Generate 8 fixed seaborn colors
    # --------------------------------
    palette = sns.color_palette("colorblind", 8)  # ← beauty + consistency

    model_names = [
        "flux",
        "Proteus",
        "sd3",
        "sana",
        "hunyuan",
        "qwen",
        "kolors",
        "Wan2.1"
    ]

    # bind each model to one color permanently
    fixed_colors = {model: palette[i] for i, model in enumerate(model_names)}

    # ------------------------------
    # Load data
    # ------------------------------
    base_path = "/content/bias/attract"
    data = {}

    for model in model_names:
        json_path = os.path.join(base_path, f"{model}_attractiveness_counts.json")
        with open(json_path, "r") as f:
            data[model] = json.load(f)["person"]

    # attractiveness categories
    attr_groups = list(data[model_names[0]].keys())
    n_attr = len(attr_groups)

    # proportions
    proportions = {}
    for model, counts in data.items():
        total = sum(counts.values())
        proportions[model] = [counts[a] / total for a in attr_groups]

    # ------------------------------
    # Plot
    # ------------------------------
    x = np.arange(n_attr)
    bar_width = 0.8 / len(model_names)

    plt.figure(figsize=(10, 6))

    # draw bars with fixed seaborn colors
    for i, model in enumerate(model_names):
        plt.bar(
            x + i * bar_width,
            proportions[model],
            width=bar_width,
            color=fixed_colors[model],  # ★ fixed seaborn color
            label=model
        )

    plt.xticks(x + bar_width * len(model_names) / 2, attr_groups, fontsize=12)
    plt.ylabel("Proportion", fontsize=12)
    plt.title("Neutral Attractiveness Distribution Across Models", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    # ------------------------------
    # Legend (inside but non-blocking)
    # ------------------------------
    handles, labels = plt.gca().get_legend_handles_labels()
    ordered_handles = [handles[labels.index(name)] for name in model_names]

    plt.legend(
        ordered_handles,
        model_names,
        fontsize=10,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.82, 1),  # place above plot so it never blocks bars
        framealpha=0.9
    )

    plt.tight_layout()
    plt.savefig("neutral_attractiveness_distribution_v2.png", dpi=300, bbox_inches="tight")
    plt.show()


def real_world_gender():
    # -------------------------------------
    # Real-world gender distribution (from UN)
    # -------------------------------------
    world_gender = {
        "Male": 0.5028,
        "Female": 0.4972
    }

    gender_keys = ["Male", "Female"]
    P_world = np.array([world_gender[k] for k in gender_keys])

    # -------------------------------------
    # Divergence functions
    # -------------------------------------
    def kl_divergence(p, q, eps=1e-12):
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return np.sum(p * np.log(p / q))

    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    def tv_distance(p, q):
        return 0.5 * np.sum(np.abs(p - q))

    # -------------------------------------
    # Define model list
    # -------------------------------------
    model_names = [
        "flux",
        "Proteus",
        "sd3",
        "sana",
        "hunyuan",
        "qwen",
        "kolors",
        "Wan2.1"
    ]

    base_path = "/content/bias/gender"

    def json_path(model):
        return os.path.join(base_path, f"{model}_gender_counts.json")

    # -------------------------------------
    # Compute divergences
    # -------------------------------------
    results = {}

    for model in model_names:
        path = json_path(model)

        with open(path, "r") as f:
            data = json.load(f)

        # Extract neutral/person distribution
        counts = data["person"]
        total = sum(counts.values())
        P_model = np.array([counts[k] / total for k in gender_keys])

        results[model] = {
            "KL": kl_divergence(P_world, P_model),
            "JS": js_divergence(P_world, P_model),
            "TVD": tv_distance(P_world, P_model),
            "model_dist": P_model
        }

    # -------------------------------------
    # Create DataFrame
    # -------------------------------------
    df_gender = pd.DataFrame(results).T
    print(df_gender)

def real_world_race():
    # ---------------------------------------------------
    # 1. Real-world race distribution (normalized)
    # ---------------------------------------------------
    race_keys = ["White", "Black", "Asian", "Others"]

    world_raw = {
        "Asian": 29.04,
        "Others": 36.10,
        "White": 9.67,
        "Black": 19.21
    }

    # Normalize to sum to 1
    total = sum(world_raw.values())
    world_race = {k: world_raw[k] / total for k in world_raw}

    P_world = np.array([world_race[k] for k in race_keys])

    print("Real world race distribution (normalized):")
    print(P_world)

    # ---------------------------------------------------
    # 2. Divergence functions
    # ---------------------------------------------------
    def kl_divergence(p, q, eps=1e-12):
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return np.sum(p * np.log(p / q))

    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    def tv_distance(p, q):
        return 0.5 * np.sum(np.abs(p - q))

    # ---------------------------------------------------
    # 3. Model list
    # ---------------------------------------------------
    model_names = [
        "flux",
        "Proteus",
        "sd3",
        "sana",
        "hunyuan",
        "qwen",
        "kolors",
        "Wan2.1"
    ]

    base_path = "/content/bias/race"

    def json_path(model):
        return os.path.join(base_path, f"{model}_emotion_race_counts.json")

    # ---------------------------------------------------
    # 4. Compute divergences
    # ---------------------------------------------------
    results = {}

    for model in model_names:
        path = json_path(model)

        with open(path, "r") as f:
            data = json.load(f)

        # extract person race counts
        counts = data["person"]
        total_count = sum(counts.values())
        P_model = np.array([counts[k] / total_count for k in race_keys])

        results[model] = {
            "KL": kl_divergence(P_world, P_model),
            "JS": js_divergence(P_world, P_model),
            "TVD": tv_distance(P_world, P_model),
            "model_dist": P_model
        }

    # ---------------------------------------------------
    # 5. Output DataFrame
    # ---------------------------------------------------
    df_race = pd.DataFrame(results).T
    print("\nRace divergence results:")
    print(df_race)


if __name__ == '__main__':
    # age_distribution2()
    # gender_distribution2()
    race_distribution2()
    # gender_distribution()
    # race_distribution()