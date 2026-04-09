import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json



def compute_ahead():
    # 读取数据
    df = pd.read_csv("/Users/wmt/projects/bias/data/aheadAttrRatings.csv")

    # 过滤 beautification == original
    df_original = df[df["beautification"] == "original"]

    # 统计 actual_label 分布
    label_counts = df_original["actual_label"].value_counts().sort_index()

    label_map = {0: "low", 1: "medium", 2: "high"}
    labels = [label_map[i] for i in label_counts.index]

    # 画分布图
    plt.figure(figsize=(6,4))
    plt.bar(labels, label_counts.values)
    plt.xlabel("Actual Label")
    plt.ylabel("Count")
    plt.title("Distribution of actual_label (beautification = original)")
    plt.show()

    print(label_counts)


def compute_western_models():
    model_files = {
        "Flux": "/Users/wmt/projects/bias/data/jsons/attract/flux_attractiveness_counts.json",
        "Proteus": "/Users/wmt/projects/bias/data/jsons/attract/Proteus_attractiveness_counts.json",
        "SANA": "/Users/wmt/projects/bias/data/jsons/attract/sana_attractiveness_counts.json",
        "SD3": "/Users/wmt/projects/bias/data/jsons/attract/sd3_attractiveness_counts.json",
    }

    model_colors = {
        "Flux": "#08306B",
        "Proteus": "#2171B5",
        "SANA": "#6BAED6",
        "SD3": "#C6DBEF",
        "Hunyuan": "#7F2704",
        "Kolors": "#A63603",
        "Qwen": "#E6550D",
        "Wan2.1": "#FDBE85",
    }

    labels = ["low", "medium", "high"]

    # ============================
    # 2. Load JSON and normalize
    # ============================
    probs = {}

    for model, path in model_files.items():
        with open(path, "r") as f:
            js = json.load(f)

        counts = np.array([js["person"][k] for k in labels])
        probs[model] = counts / counts.sum()

    # ============================
    # 3. Plot grouped bars
    # ============================
    num_models = len(model_files)
    bar_width = 0.18

    x = np.arange(len(labels))  # low, medium, high

    plt.figure(figsize=(8, 4))

    for i, model in enumerate(model_files.keys()):
        plt.bar(
            x + i * bar_width,
            probs[model],
            width=bar_width,
            color=model_colors[model],
            label=model
        )

    # ============================
    # 4. Formatting
    # ============================
    plt.xticks(x + bar_width * (num_models - 1) / 2, labels)
    plt.ylabel("Proportion")
    plt.xlabel("Attractiveness")
    plt.title("Attractiveness Distribution of Generated Faces")
    plt.legend(frameon=False, ncol=4)

    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("attractiveness_distribution_models.png", dpi=300)
    plt.show()



def attractiveness_distribution():
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

    # --------------------------------
    # 2. Colors
    # --------------------------------
    model_colors = {
        "flux": "#08306B",
        "Proteus": "#2171B5",
        "sana": "#6BAED6",
        "sd3": "#C6DBEF",
        "hunyuan": "#7F2704",
        "kolors": "#A63603",
        "qwen": "#E6550D",
        "Wan2.1": "#FDBE85",
    }

    # --------------------------------
    # 3. Load data
    # --------------------------------
    base_path = "/Users/wmt/projects/bias/data/jsons/attract"

    data = {}
    for model in model_names:
        with open(os.path.join(base_path, f"{model}_attractiveness_counts.json"), "r") as f:
            data[model] = json.load(f)["person"]

    attr_groups = ["low", "medium", "high"]
    n_attr = len(attr_groups)

    # --------------------------------
    # 4. Convert to proportions
    # --------------------------------
    proportions = {}
    for model, counts in data.items():
        total = sum(counts.values())
        proportions[model] = [counts[g] / total for g in attr_groups]

    # --------------------------------
    # 5. Baseline (from your GT statistics)
    # --------------------------------
    # aheadAttrRatings.csv -> beautification == original
    real_world_attr = {
        "low": 0.474,
        "medium": 0.520,
        "high": 0.006
    }
    real_values = [real_world_attr[g] for g in attr_groups]

    # --------------------------------
    # 6. Layout parameters (same as age)
    # --------------------------------
    bar_width = 0.09
    intra_group_gap = bar_width * 1.7
    inter_block_gap = bar_width * 5.0

    block_width = bar_width * 8 + intra_group_gap
    step = block_width + inter_block_gap
    x = np.arange(n_attr) * step

    def get_x_pos(attr_idx, model):
        base = x[attr_idx]
        if model in western_models:
            i = western_models.index(model)
            return base + i * bar_width
        else:
            i = chinese_models.index(model)
            return base + len(western_models) * bar_width + intra_group_gap + i * bar_width

    # --------------------------------
    # 7. Plot
    # --------------------------------
    plt.figure(figsize=(10, 6))

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

    # Bars
    for model in model_names:
        positions = [get_x_pos(i, model) for i in range(n_attr)]
        plt.bar(
            positions,
            proportions[model],
            width=bar_width,
            color=model_colors[model],
            label=model
        )

    # Baseline dashed lines
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
    # 8. Axis fonts
    # --------------------------------
    plt.xticks(x + block_width / 2, attr_groups, fontsize=17)
    plt.ylabel("Proportion", fontsize=14)
    plt.yticks(fontsize=17)

    # --------------------------------
    # 9. Legend
    # --------------------------------
    handles, labels = plt.gca().get_legend_handles_labels()
    ordered_handles = [handles[labels.index(m)] for m in model_names]
    ordered_labels = [legend_name_map[m] for m in model_names]

    plt.legend(
        ordered_handles,
        ordered_labels,
        ncol=2,
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.8, 1),
        framealpha=0.9,
        handlelength=1.8,
        handleheight=1.0,
        borderpad=0.6
    )

    plt.tight_layout()
    plt.savefig("neutral_attractiveness_distribution_final.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    # compute_ahead()
    # compute_western_models()
    attractiveness_distribution()