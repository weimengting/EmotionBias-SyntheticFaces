import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from scipy.spatial.distance import jensenshannon

def western_race():

    # =====================================================
    # 0. Model files
    # =====================================================
    model_files = {
        "FLUX-Schnell": {
            "sad": "data/res/flux/sad_res.csv",
            "unhappy": "data/res/flux/unhappy_res.csv",
        },
        "Proteus": {
            "sad": "data/res/proteus/sad_res.csv",
            "unhappy": "data/res/proteus/unhappy_res.csv",
        },
        "SANA": {
            "sad": "data/res/sana/sad_res.csv",
            "unhappy": "data/res/sana/unhappy_res.csv",
        },
        "SD3": {
            "sad": "data/res/sd3/sad_res.csv",
            "unhappy": "data/res/sd3/unhappy_res.csv",
        },
    }

    # =====================================================
    # Model colors (color = model)
    # =====================================================
    model_colors = {
        "FLUX-Schnell": "#08306B",
        "Proteus": "#2171B5",
        "SANA": "#6BAED6",
        "SD3": "#C6DBEF",
    }

    race_order = ["white", "black", "asian", "others"]

    race_order_raw = [
        "White", "Black", "Latino_Hispanic",
        "East Asian", "Southeast Asian",
        "Indian", "Middle Eastern"
    ]

    # =====================================================
    # 1. Seaborn style
    # =====================================================
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2
    )

    # =====================================================
    # 2. Race processing logic (与你给的完全一致)
    # =====================================================
    def parse_race_scores(x):
        x = str(x).replace("[", "").replace("]", "")
        return np.array([float(v) for v in x.split()])

    def process_race(row):
        race = row["race"]
        scores = parse_race_scores(row["race_scores_fair"])

        # Middle Eastern: 用第二高 score 映射
        if race == "Middle Eastern":
            idx = np.argsort(scores)[::-1][1]
            race = race_order_raw[idx]

        if race in ["Indian", "Latino_Hispanic"]:
            return "others"
        if race in ["East Asian", "Southeast Asian"]:
            return "asian"

        return race.lower()

    def compute_race_distribution(csv_path):
        df = pd.read_csv(csv_path)

        # 只统计 face0
        df = df[df["face_name_align"].str.contains("face0")].copy()

        df["race_group"] = df.apply(process_race, axis=1)

        return (
            df["race_group"]
            .value_counts(normalize=True)
            .reindex(race_order, fill_value=0.0)
        )

    # =====================================================
    # 3. Compute distributions (Sad / Unhappy)
    # =====================================================
    results = {}

    for model, paths in model_files.items():
        results[(model, "Sad")] = compute_race_distribution(paths["sad"])
        results[(model, "Unhappy")] = compute_race_distribution(paths["unhappy"])

    # =====================================================
    # 4. Convert to tidy DataFrame for seaborn
    # =====================================================
    rows = []
    for model in model_files.keys():
        for emotion in ["Sad", "Unhappy"]:
            dist = results[(model, emotion)]
            for race, value in dist.items():
                rows.append({
                    "Race": race.capitalize(),
                    "Proportion": value,
                    "Model": model,
                    "Emotion": emotion,
                    "ModelEmotion": f"{model}_{emotion}",  # 👈 关键
                })

    plot_df = pd.DataFrame(rows)

    plot_df["Race"] = pd.Categorical(
        plot_df["Race"],
        categories=[r.capitalize() for r in race_order],
        ordered=True
    )

    # =====================================================
    # 2. 为 seaborn 准备 palette（颜色只由 Model 决定）
    # =====================================================
    palette = {}
    for model in model_files.keys():
        palette[f"{model}_Sad"] = model_colors[model]
        palette[f"{model}_Unhappy"] = model_colors[model]

    # =====================================================
    # 3. Seaborn barplot（现在真的会画 6 根）
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=plot_df,
        x="Race",
        y="Proportion",
        hue="ModelEmotion",
        palette=palette,
        dodge=True,
        errorbar=None,
        ax=ax,
    )

    # =====================================================
    # 4. 设置 hatch：只给 Unhappy
    # =====================================================
    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        if row["Emotion"] == "Unhappy":
            patch.set_hatch("//")

    # =====================================================
    # 5. Axis & title
    # =====================================================
    ax.set_ylabel("Proportion")
    ax.set_xlabel("")
    ax.set_title("Race distribution under Sad vs Unhappy prompts")



    # y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Percentage (%)")
    for patch in ax.patches:
        height = patch.get_height()
        if height <= 0:
            continue

        ax.annotate(
            f"{int(round(height * 100))}",
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # =====================================================
    # 6. Legend（拆成 Model / Emotion，论文级）
    # =====================================================
    ax.legend_.remove()

    model_legend = [
        Patch(facecolor=model_colors[m], edgecolor="black", label=m)
        for m in model_files.keys()
    ]

    emotion_legend = [
        Patch(facecolor="white", edgecolor="black", label="Sad"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Unhappy"),
    ]

    leg1 = ax.legend(
        handles=model_legend,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(0.7, 1.0),  # 👈 右侧外面
        borderaxespad=0,
        frameon=True
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=emotion_legend,
        title="Emotion",
        loc="upper center",
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("sad_vs_unhappy_race_distribution_western.png", dpi=300)
    plt.show()


def chinese_race():

    # =====================================================
    # 0. Model files
    # =====================================================
    model_files = {
        "Hunyuan": {
            "sad": "data/res/hunyuan/sad_res.csv",
            "unhappy": "data/res/hunyuan/unhappy_res.csv",
        },
        "Kolors": {
            "sad": "data/res/kolors/sad_res.csv",
            "unhappy": "data/res/kolors/unhappy_res.csv",
        },
        "Qwen": {
            "sad": "data/res/qwen/sad_res.csv",
            "unhappy": "data/res/qwen/unhappy_res.csv",
        },
        "Wan2.1": {
            "sad": "data/res/wan2.1/sad_res.csv",
            "unhappy": "data/res/wan2.1/unhappy_res.csv",
        },
    }

    # =====================================================
    # Model colors (color = model)
    # =====================================================
    model_colors = {
        "Hunyuan": "#7F2704",  # deep orange-brown
        "Kolors": "#A63603",
        "Qwen": "#E6550D",
        "Wan2.1": "#FDBE85",
    }

    race_order = ["white", "black", "asian", "others"]

    race_order_raw = [
        "White", "Black", "Latino_Hispanic",
        "East Asian", "Southeast Asian",
        "Indian", "Middle Eastern"
    ]

    # =====================================================
    # 1. Seaborn style
    # =====================================================
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2
    )

    # =====================================================
    # 2. Race processing logic (与你给的完全一致)
    # =====================================================
    def parse_race_scores(x):
        x = str(x).replace("[", "").replace("]", "")
        return np.array([float(v) for v in x.split()])

    def process_race(row):
        race = row["race"]
        scores = parse_race_scores(row["race_scores_fair"])

        # Middle Eastern: 用第二高 score 映射
        if race == "Middle Eastern":
            idx = np.argsort(scores)[::-1][1]
            race = race_order_raw[idx]

        if race in ["Indian", "Latino_Hispanic"]:
            return "others"
        if race in ["East Asian", "Southeast Asian"]:
            return "asian"

        return race.lower()

    def compute_race_distribution(csv_path):
        df = pd.read_csv(csv_path)

        # 只统计 face0
        df = df[df["face_name_align"].str.contains("face0")].copy()

        df["race_group"] = df.apply(process_race, axis=1)

        return (
            df["race_group"]
            .value_counts(normalize=True)
            .reindex(race_order, fill_value=0.0)
        )

    # =====================================================
    # 3. Compute distributions (Sad / Unhappy)
    # =====================================================
    results = {}

    for model, paths in model_files.items():
        results[(model, "Sad")] = compute_race_distribution(paths["sad"])
        results[(model, "Unhappy")] = compute_race_distribution(paths["unhappy"])

    # =====================================================
    # 4. Convert to tidy DataFrame for seaborn
    # =====================================================
    rows = []
    for model in model_files.keys():
        for emotion in ["Sad", "Unhappy"]:
            dist = results[(model, emotion)]
            for race, value in dist.items():
                rows.append({
                    "Race": race.capitalize(),
                    "Proportion": value,
                    "Model": model,
                    "Emotion": emotion,
                    "ModelEmotion": f"{model}_{emotion}",  # 👈 关键
                })

    plot_df = pd.DataFrame(rows)

    plot_df["Race"] = pd.Categorical(
        plot_df["Race"],
        categories=[r.capitalize() for r in race_order],
        ordered=True
    )

    # =====================================================
    # 2. 为 seaborn 准备 palette（颜色只由 Model 决定）
    # =====================================================
    palette = {}
    for model in model_files.keys():
        palette[f"{model}_Sad"] = model_colors[model]
        palette[f"{model}_Unhappy"] = model_colors[model]

    # =====================================================
    # 3. Seaborn barplot（现在真的会画 6 根）
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=plot_df,
        x="Race",
        y="Proportion",
        hue="ModelEmotion",
        palette=palette,
        dodge=True,
        errorbar=None,
        ax=ax,
    )

    # =====================================================
    # 4. 设置 hatch：只给 Unhappy
    # =====================================================
    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        if row["Emotion"] == "Unhappy":
            patch.set_hatch("//")

    # =====================================================
    # 5. Axis & title
    # =====================================================
    ax.set_ylabel("Proportion")
    ax.set_xlabel("")
    ax.set_title("Race distribution under Sad vs Unhappy prompts")



    # y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Percentage (%)")
    for patch in ax.patches:
        height = patch.get_height()
        if height <= 0:
            continue

        ax.annotate(
            f"{int(round(height * 100))}",
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # =====================================================
    # 6. Legend（拆成 Model / Emotion，论文级）
    # =====================================================
    ax.legend_.remove()

    model_legend = [
        Patch(facecolor=model_colors[m], edgecolor="black", label=m)
        for m in model_files.keys()
    ]

    emotion_legend = [
        Patch(facecolor="white", edgecolor="black", label="Sad"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Unhappy"),
    ]

    leg1 = ax.legend(
        handles=model_legend,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(0.7, 1.0),  # 👈 右侧外面
        borderaxespad=0,
        frameon=True
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=emotion_legend,
        title="Emotion",
        loc="upper center",
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("sad_vs_unhappy_race_distribution_chinese.png", dpi=300)
    plt.show()


def western_age():


    # =====================================================
    # 0. Model files
    # =====================================================
    model_files = {
        "FLUX-Schnell": {
            "sad": "data/res/flux/sad_res.csv",
            "unhappy": "data/res/flux/unhappy_res.csv",
        },
        "Proteus": {
            "sad": "data/res/proteus/sad_res.csv",
            "unhappy": "data/res/proteus/unhappy_res.csv",
        },
        "SANA": {
            "sad": "data/res/sana/sad_res.csv",
            "unhappy": "data/res/sana/unhappy_res.csv",
        },
        "SD3": {
            "sad": "data/res/sd3/sad_res.csv",
            "unhappy": "data/res/sd3/unhappy_res.csv",
        },
    }

    # =====================================================
    # Model colors
    # =====================================================
    model_colors = {
        "FLUX-Schnell": "#08306B",
        "Proteus": "#2171B5",
        "SANA": "#6BAED6",
        "SD3": "#C6DBEF",
    }

    age_order = ["0-9", "10-19", "20-39", "40-59", "60+"]

    # =====================================================
    # 1. Seaborn style
    # =====================================================
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2
    )

    # =====================================================
    # 2. Age processing logic（按你给的映射）
    # =====================================================
    def map_age(age):
        age = str(age)

        if age in ["0-2", "3-9"]:
            return "0-9"
        if age == "10-19":
            return "10-19"
        if age in ["20-29", "30-39"]:
            return "20-39"
        if age in ["40-49", "50-59"]:
            return "40-59"
        if age in ["60-69", "70+"]:
            return "60+"

        return None  # 安全兜底

    def compute_age_distribution(csv_path):
        df = pd.read_csv(csv_path)

        # 只统计 face0
        df = df[df["face_name_align"].str.contains("face0")].copy()

        df["age_group"] = df["age"].apply(map_age)

        return (
            df["age_group"]
            .value_counts(normalize=True)
            .reindex(age_order, fill_value=0.0)
        )

    # =====================================================
    # 3. Compute distributions (Sad / Unhappy)
    # =====================================================
    results = {}

    for model, paths in model_files.items():
        results[(model, "Sad")] = compute_age_distribution(paths["sad"])
        results[(model, "Unhappy")] = compute_age_distribution(paths["unhappy"])

    # =====================================================
    # 4. Convert to tidy DataFrame for seaborn
    # =====================================================
    rows = []
    for model in model_files.keys():
        for emotion in ["Sad", "Unhappy"]:
            dist = results[(model, emotion)]
            for age, value in dist.items():
                rows.append({
                    "Age": age,
                    "Proportion": value,
                    "Model": model,
                    "Emotion": emotion,
                    "ModelEmotion": f"{model}_{emotion}",
                })

    plot_df = pd.DataFrame(rows)

    plot_df["Age"] = pd.Categorical(
        plot_df["Age"],
        categories=age_order,
        ordered=True
    )

    # =====================================================
    # 5. Palette（颜色只由 Model 决定）
    # =====================================================
    palette = {}
    for model in model_files.keys():
        palette[f"{model}_Sad"] = model_colors[model]
        palette[f"{model}_Unhappy"] = model_colors[model]

    # =====================================================
    # 6. Plot (seaborn)
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=plot_df,
        x="Age",
        y="Proportion",
        hue="ModelEmotion",
        palette=palette,
        dodge=True,
        errorbar=None,
        ax=ax,
    )

    # Hatch for Unhappy
    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        if row["Emotion"] == "Unhappy":
            patch.set_hatch("//")

    # =====================================================
    # 7. Axis & percentage
    # =====================================================
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.set_title("Age distribution under Sad vs Unhappy prompts")

    # Value labels
    for patch in ax.patches:
        height = patch.get_height()
        if height <= 0:
            continue
        ax.annotate(
            f"{int(round(height * 100))}",
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # =====================================================
    # 8. Legend（同 race 图，完全一致）
    # =====================================================
    ax.legend_.remove()

    model_legend = [
        Patch(facecolor=model_colors[m], edgecolor="black", label=m)
        for m in model_files.keys()
    ]

    emotion_legend = [
        Patch(facecolor="white", edgecolor="black", label="Sad"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Unhappy"),
    ]

    leg1 = ax.legend(
        handles=model_legend,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0,
        frameon=True
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=emotion_legend,
        title="Emotion",
        loc="upper left",
        bbox_to_anchor=(0.2, 1.0),  # 👈 你可以随便改这里
        borderaxespad=0,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("sad_vs_unhappy_age_distribution_western.png", dpi=300)
    plt.show()

def chinese_age():


    # =====================================================
    # 0. Model files
    # =====================================================
    model_files = {
        "Hunyuan": {
            "sad": "data/res/hunyuan/sad_res.csv",
            "unhappy": "data/res/hunyuan/unhappy_res.csv",
        },
        "Kolors": {
            "sad": "data/res/kolors/sad_res.csv",
            "unhappy": "data/res/kolors/unhappy_res.csv",
        },
        "Qwen": {
            "sad": "data/res/qwen/sad_res.csv",
            "unhappy": "data/res/qwen/unhappy_res.csv",
        },
        "Wan2.1": {
            "sad": "data/res/wan2.1/sad_res.csv",
            "unhappy": "data/res/wan2.1/unhappy_res.csv",
        },
    }

    # =====================================================
    # Model colors (color = model)
    # =====================================================
    model_colors = {
        "Hunyuan": "#7F2704",  # deep orange-brown
        "Kolors": "#A63603",
        "Qwen": "#E6550D",
        "Wan2.1": "#FDBE85",
    }

    age_order = ["0-9", "10-19", "20-39", "40-59", "60+"]

    # =====================================================
    # 1. Seaborn style
    # =====================================================
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2
    )

    # =====================================================
    # 2. Age processing logic（按你给的映射）
    # =====================================================
    def map_age(age):
        age = str(age)

        if age in ["0-2", "3-9"]:
            return "0-9"
        if age == "10-19":
            return "10-19"
        if age in ["20-29", "30-39"]:
            return "20-39"
        if age in ["40-49", "50-59"]:
            return "40-59"
        if age in ["60-69", "70+"]:
            return "60+"

        return None  # 安全兜底

    def compute_age_distribution(csv_path):
        df = pd.read_csv(csv_path)

        # 只统计 face0
        df = df[df["face_name_align"].str.contains("face0")].copy()

        df["age_group"] = df["age"].apply(map_age)

        return (
            df["age_group"]
            .value_counts(normalize=True)
            .reindex(age_order, fill_value=0.0)
        )

    # =====================================================
    # 3. Compute distributions (Sad / Unhappy)
    # =====================================================
    results = {}

    for model, paths in model_files.items():
        results[(model, "Sad")] = compute_age_distribution(paths["sad"])
        results[(model, "Unhappy")] = compute_age_distribution(paths["unhappy"])

    # =====================================================
    # 4. Convert to tidy DataFrame for seaborn
    # =====================================================
    rows = []
    for model in model_files.keys():
        for emotion in ["Sad", "Unhappy"]:
            dist = results[(model, emotion)]
            for age, value in dist.items():
                rows.append({
                    "Age": age,
                    "Proportion": value,
                    "Model": model,
                    "Emotion": emotion,
                    "ModelEmotion": f"{model}_{emotion}",
                })

    plot_df = pd.DataFrame(rows)

    plot_df["Age"] = pd.Categorical(
        plot_df["Age"],
        categories=age_order,
        ordered=True
    )

    # =====================================================
    # 5. Palette（颜色只由 Model 决定）
    # =====================================================
    palette = {}
    for model in model_files.keys():
        palette[f"{model}_Sad"] = model_colors[model]
        palette[f"{model}_Unhappy"] = model_colors[model]

    # =====================================================
    # 6. Plot (seaborn)
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=plot_df,
        x="Age",
        y="Proportion",
        hue="ModelEmotion",
        palette=palette,
        dodge=True,
        errorbar=None,
        ax=ax,
    )

    # Hatch for Unhappy
    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        if row["Emotion"] == "Unhappy":
            patch.set_hatch("//")

    # =====================================================
    # 7. Axis & percentage
    # =====================================================
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.set_title("Age distribution under Sad vs Unhappy prompts")

    # Value labels
    for patch in ax.patches:
        height = patch.get_height()
        if height <= 0:
            continue
        ax.annotate(
            f"{int(round(height * 100))}",
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # =====================================================
    # 8. Legend（同 race 图，完全一致）
    # =====================================================
    ax.legend_.remove()

    model_legend = [
        Patch(facecolor=model_colors[m], edgecolor="black", label=m)
        for m in model_files.keys()
    ]

    emotion_legend = [
        Patch(facecolor="white", edgecolor="black", label="Sad"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Unhappy"),
    ]

    leg1 = ax.legend(
        handles=model_legend,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0,
        frameon=True
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=emotion_legend,
        title="Emotion",
        loc="upper left",
        bbox_to_anchor=(0.2, 1.0),  # 👈 你可以随便改这里
        borderaxespad=0,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("sad_vs_unhappy_age_distribution_chinese.png", dpi=300)
    plt.show()


def western_gender():
    model_files = {
        "FLUX-Schnell": {
            "sad": "data/res/flux/sad_res.csv",
            "unhappy": "data/res/flux/unhappy_res.csv",
        },
        "Proteus": {
            "sad": "data/res/proteus/sad_res.csv",
            "unhappy": "data/res/proteus/unhappy_res.csv",
        },
        "SANA": {
            "sad": "data/res/sana/sad_res.csv",
            "unhappy": "data/res/sana/unhappy_res.csv",
        },
        "SD3": {
            "sad": "data/res/sd3/sad_res.csv",
            "unhappy": "data/res/sd3/unhappy_res.csv",
        },
    }

    # =====================================================
    # Model colors
    # =====================================================
    model_colors = {
        "FLUX-Schnell": "#08306B",
        "Proteus": "#2171B5",
        "SANA": "#6BAED6",
        "SD3": "#C6DBEF",
    }

    gender_order = ["Male", "Female"]

    # =====================================================
    # 1. Seaborn style
    # =====================================================
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2
    )

    # =====================================================
    # 2. Gender processing logic
    # =====================================================
    def compute_gender_distribution(csv_path):
        df = pd.read_csv(csv_path)

        # 只统计 face0
        df = df[df["face_name_align"].str.contains("face0")].copy()

        return (
            df["gender"]
            .value_counts(normalize=True)
            .reindex(gender_order, fill_value=0.0)
        )

    # =====================================================
    # 3. Compute distributions (Sad / Unhappy)
    # =====================================================
    results = {}

    for model, paths in model_files.items():
        results[(model, "Sad")] = compute_gender_distribution(paths["sad"])
        results[(model, "Unhappy")] = compute_gender_distribution(paths["unhappy"])

    # =====================================================
    # 4. Convert to tidy DataFrame for seaborn
    # =====================================================
    rows = []
    for model in model_files.keys():
        for emotion in ["Sad", "Unhappy"]:
            dist = results[(model, emotion)]
            for gender, value in dist.items():
                rows.append({
                    "Gender": gender,
                    "Proportion": value,
                    "Model": model,
                    "Emotion": emotion,
                    "ModelEmotion": f"{model}_{emotion}",
                })

    plot_df = pd.DataFrame(rows)

    plot_df["Gender"] = pd.Categorical(
        plot_df["Gender"],
        categories=gender_order,
        ordered=True
    )

    # =====================================================
    # 5. Palette（颜色只由 Model 决定）
    # =====================================================
    palette = {}
    for model in model_files.keys():
        palette[f"{model}_Sad"] = model_colors[model]
        palette[f"{model}_Unhappy"] = model_colors[model]

    # =====================================================
    # 6. Plot (seaborn)
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=plot_df,
        x="Gender",
        y="Proportion",
        hue="ModelEmotion",
        palette=palette,
        dodge=True,
        errorbar=None,
        ax=ax,
    )

    # Hatch for Unhappy
    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        if row["Emotion"] == "Unhappy":
            patch.set_hatch("//")

    # =====================================================
    # 7. Axis & percentage
    # =====================================================
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.set_title("Gender distribution under Sad vs Unhappy prompts")

    # Value labels
    for patch in ax.patches:
        height = patch.get_height()
        if height <= 0:
            continue
        ax.annotate(
            f"{int(round(height * 100))}",
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # =====================================================
    # 8. Legend（与 race / age 完全一致）
    # =====================================================
    ax.legend_.remove()

    model_legend = [
        Patch(facecolor=model_colors[m], edgecolor="black", label=m)
        for m in model_files.keys()
    ]

    emotion_legend = [
        Patch(facecolor="white", edgecolor="black", label="Sad"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Unhappy"),
    ]

    leg1 = ax.legend(
        handles=model_legend,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(0.58, 1.0),
        borderaxespad=0,
        frameon=True
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=emotion_legend,
        title="Emotion",
        loc="upper left",
        bbox_to_anchor=(0.45, 1.0),  # 👈 位置你可以自由调
        borderaxespad=0,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("sad_vs_unhappy_gender_distribution_western.png", dpi=300)
    plt.show()


def chinese_gender():
    model_files = {
        "Hunyuan": {
            "sad": "data/res/hunyuan/sad_res.csv",
            "unhappy": "data/res/hunyuan/unhappy_res.csv",
        },
        "Kolors": {
            "sad": "data/res/kolors/sad_res.csv",
            "unhappy": "data/res/kolors/unhappy_res.csv",
        },
        "Qwen": {
            "sad": "data/res/qwen/sad_res.csv",
            "unhappy": "data/res/qwen/unhappy_res.csv",
        },
        "Wan2.1": {
            "sad": "data/res/wan2.1/sad_res.csv",
            "unhappy": "data/res/wan2.1/unhappy_res.csv",
        },
    }

    # =====================================================
    # Model colors (color = model)
    # =====================================================
    model_colors = {
        "Hunyuan": "#7F2704",  # deep orange-brown
        "Kolors": "#A63603",
        "Qwen": "#E6550D",
        "Wan2.1": "#FDBE85",
    }

    gender_order = ["Male", "Female"]

    # =====================================================
    # 1. Seaborn style
    # =====================================================
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2
    )

    # =====================================================
    # 2. Gender processing logic
    # =====================================================
    def compute_gender_distribution(csv_path):
        df = pd.read_csv(csv_path)

        # 只统计 face0
        df = df[df["face_name_align"].str.contains("face0")].copy()

        return (
            df["gender"]
            .value_counts(normalize=True)
            .reindex(gender_order, fill_value=0.0)
        )

    # =====================================================
    # 3. Compute distributions (Sad / Unhappy)
    # =====================================================
    results = {}

    for model, paths in model_files.items():
        results[(model, "Sad")] = compute_gender_distribution(paths["sad"])
        results[(model, "Unhappy")] = compute_gender_distribution(paths["unhappy"])

    # =====================================================
    # 4. Convert to tidy DataFrame for seaborn
    # =====================================================
    rows = []
    for model in model_files.keys():
        for emotion in ["Sad", "Unhappy"]:
            dist = results[(model, emotion)]
            for gender, value in dist.items():
                rows.append({
                    "Gender": gender,
                    "Proportion": value,
                    "Model": model,
                    "Emotion": emotion,
                    "ModelEmotion": f"{model}_{emotion}",
                })

    plot_df = pd.DataFrame(rows)

    plot_df["Gender"] = pd.Categorical(
        plot_df["Gender"],
        categories=gender_order,
        ordered=True
    )

    # =====================================================
    # 5. Palette（颜色只由 Model 决定）
    # =====================================================
    palette = {}
    for model in model_files.keys():
        palette[f"{model}_Sad"] = model_colors[model]
        palette[f"{model}_Unhappy"] = model_colors[model]

    # =====================================================
    # 6. Plot (seaborn)
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=plot_df,
        x="Gender",
        y="Proportion",
        hue="ModelEmotion",
        palette=palette,
        dodge=True,
        errorbar=None,
        ax=ax,
    )

    # Hatch for Unhappy
    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        if row["Emotion"] == "Unhappy":
            patch.set_hatch("//")

    # =====================================================
    # 7. Axis & percentage
    # =====================================================
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.set_title("Gender distribution under Sad vs Unhappy prompts")

    # Value labels
    for patch in ax.patches:
        height = patch.get_height()
        if height <= 0:
            continue
        ax.annotate(
            f"{int(round(height * 100))}",
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # =====================================================
    # 8. Legend（与 race / age 完全一致）
    # =====================================================
    ax.legend_.remove()

    model_legend = [
        Patch(facecolor=model_colors[m], edgecolor="black", label=m)
        for m in model_files.keys()
    ]

    emotion_legend = [
        Patch(facecolor="white", edgecolor="black", label="Sad"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Unhappy"),
    ]

    leg1 = ax.legend(
        handles=model_legend,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(0.58, 1.0),
        borderaxespad=0,
        frameon=True
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=emotion_legend,
        title="Emotion",
        loc="upper left",
        bbox_to_anchor=(0.45, 1.0),  # 👈 位置你可以自由调
        borderaxespad=0,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("sad_vs_unhappy_gender_distribution_chinese.png", dpi=300)
    plt.show()


def js_computation():
    # =====================================================
    # 0. Model files（与你前面一致）
    # =====================================================
    model_files = {
        "FLUX-Schnell": {
            "sad": "data/res/flux/sad_res.csv",
            "unhappy": "data/res/flux/unhappy_res.csv",
        },
        "Proteus": {
            "sad": "data/res/proteus/sad_res.csv",
            "unhappy": "data/res/proteus/unhappy_res.csv",
        },
        "SANA": {
            "sad": "data/res/sana/sad_res.csv",
            "unhappy": "data/res/sana/unhappy_res.csv",
        },
        "SD3": {
            "sad": "data/res/sd3/sad_res.csv",
            "unhappy": "data/res/sd3/unhappy_res.csv",
        },
        "Hunyuan": {
            "sad": "data/res/hunyuan/sad_res.csv",
            "unhappy": "data/res/hunyuan/unhappy_res.csv",
        },
        "Kolors": {
            "sad": "data/res/kolors/sad_res.csv",
            "unhappy": "data/res/kolors/unhappy_res.csv",
        },
        "Qwen": {
            "sad": "data/res/qwen/sad_res.csv",
            "unhappy": "data/res/qwen/unhappy_res.csv",
        },
        "Wan2.1": {
            "sad": "data/res/wan2.1/sad_res.csv",
            "unhappy": "data/res/wan2.1/unhappy_res.csv",
        }
    }

    # =====================================================
    # 1. Shared demographic processing logic
    # =====================================================

    # ---- Age mapping ----
    def map_age(age):
        age = str(age)
        if age in ["0-2", "3-9"]:
            return "0-9"
        if age == "10-19":
            return "10-19"
        if age in ["20-29", "30-39"]:
            return "20-39"
        if age in ["40-49", "50-59"]:
            return "40-59"
        if age in ["60-69", "70+"]:
            return "60+"
        return None

    age_order = ["0-9", "10-19", "20-39", "40-59", "60+"]

    # ---- Gender ----
    gender_order = ["Male", "Female"]

    # ---- Race mapping（与你之前完全一致）----
    race_order_raw = [
        "White", "Black", "Latino_Hispanic",
        "East Asian", "Southeast Asian",
        "Indian", "Middle Eastern"
    ]

    race_order = ["white", "black", "asian", "others"]

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

    # =====================================================
    # 2. Utilities
    # =====================================================
    def normalize_counts(series, index_order):
        counts = series.value_counts()
        probs = counts.reindex(index_order, fill_value=0).astype(float)
        return probs / probs.sum()

    def js_divergence(p, q):
        return jensenshannon(p, q, base=2.0) ** 2

    # =====================================================
    # 3. Compute JS table
    # =====================================================
    rows = []

    for model, paths in model_files.items():
        dfs = {}
        for emotion, path in paths.items():
            df = pd.read_csv(path)
            df = df[df["face_name_align"].str.contains("face0")].copy()

            df["age_group"] = df["age"].apply(map_age)
            df["gender_group"] = df["gender"]
            df["race_group"] = df.apply(process_race, axis=1)

            dfs[emotion] = df

        # ---- Marginal distributions ----
        js_age = js_divergence(
            normalize_counts(dfs["sad"]["age_group"], age_order),
            normalize_counts(dfs["unhappy"]["age_group"], age_order),
        )

        js_gender = js_divergence(
            normalize_counts(dfs["sad"]["gender_group"], gender_order),
            normalize_counts(dfs["unhappy"]["gender_group"], gender_order),
        )

        js_race = js_divergence(
            normalize_counts(dfs["sad"]["race_group"], race_order),
            normalize_counts(dfs["unhappy"]["race_group"], race_order),
        )

        # ---- Joint distribution ----
        def joint_distribution(df):
            joint = (
                df.groupby(["age_group", "gender_group", "race_group"])
                .size()
            )
            all_index = pd.MultiIndex.from_product(
                [age_order, gender_order, race_order],
                names=["age", "gender", "race"]
            )
            joint = joint.reindex(all_index, fill_value=0).astype(float)
            return joint / joint.sum()

        p_joint = joint_distribution(dfs["sad"])
        q_joint = joint_distribution(dfs["unhappy"])

        js_joint = js_divergence(p_joint.values, q_joint.values)

        rows.append({
            "Model": model,
            "JS (Age)": js_age,
            "JS (Gender)": js_gender,
            "JS (Race)": js_race,
            "JS (Joint)": js_joint,
        })

    # =====================================================
    # 4. Final table
    # =====================================================
    js_table = pd.DataFrame(rows)
    js_table = js_table.set_index("Model")

    print(js_table)

    # Optional: save
    js_table.to_csv("sad_vs_unhappy_js_table.csv")


def vis_western():
    TOP_K = 8

    model_files = {
        "FLUX-Schnell": {
            "sad": "data/res/flux/sad_res.csv",
            "unhappy": "data/res/flux/unhappy_res.csv",
        },
        "Proteus": {
            "sad": "data/res/proteus/sad_res.csv",
            "unhappy": "data/res/proteus/unhappy_res.csv",
        },
        "SANA": {
            "sad": "data/res/sana/sad_res.csv",
            "unhappy": "data/res/sana/unhappy_res.csv",
        },
        "SD3": {
            "sad": "data/res/sd3/sad_res.csv",
            "unhappy": "data/res/sd3/unhappy_res.csv",
        },
    }

    # =====================================================
    # 1. Demographic processing (与你之前完全一致)
    # =====================================================

    def map_age(age):
        age = str(age)
        if age in ["0-2", "3-9"]:
            return "0-9"
        if age == "10-19":
            return "10-19"
        if age in ["20-29", "30-39"]:
            return "20-39"
        if age in ["40-49", "50-59"]:
            return "40-59"
        if age in ["60-69", "70+"]:
            return "60+"
        return None

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

    # =====================================================
    # 2. Utilities
    # =====================================================
    def load_and_process(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["face_name_align"].str.contains("face0")].copy()

        df["age"] = df["age"].apply(map_age)
        df["gender"] = df["gender"]
        df["race"] = df.apply(process_race, axis=1)

        return df

    def joint_distribution(df):
        joint = (
            df.groupby(["age", "gender", "race"])
            .size()
            .reset_index(name="count")
        )
        joint["prob"] = joint["count"] / joint["count"].sum()
        return joint[["age", "gender", "race", "prob"]]

    # =====================================================
    # 3. Compute aggregated Δ across models
    # =====================================================
    all_deltas = []

    for model, paths in model_files.items():
        df_sad = load_and_process(paths["sad"])
        df_unhappy = load_and_process(paths["unhappy"])

        p_sad = joint_distribution(df_sad)
        p_unhappy = joint_distribution(df_unhappy)

        merged = pd.merge(
            p_unhappy,
            p_sad,
            on=["age", "gender", "race"],
            how="outer",
            suffixes=("_unhappy", "_sad")
        ).fillna(0)

        merged["delta"] = merged["prob_unhappy"] - merged["prob_sad"]
        merged["model"] = model

        all_deltas.append(merged)

    all_deltas = pd.concat(all_deltas, ignore_index=True)

    # 👉 对模型取平均
    agg = (
        all_deltas
        .groupby(["age", "gender", "race"], as_index=False)
        .agg(delta_mean=("delta", "mean"))
    )

    # Label for plotting
    agg["group"] = (
            agg["age"] + " · " +
            agg["gender"] + " · " +
            agg["race"].str.capitalize()
    )

    # Select Top-K by absolute mean shift
    topk = (
        agg
        .reindex(agg["delta_mean"].abs().sort_values(ascending=False).index)
        .head(TOP_K)
        .sort_values("delta_mean")
    )

    # =====================================================
    # 4. Plot: Diverging bar chart (aggregated)
    # =====================================================
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        data=topk,
        x="delta_mean",
        y="group",
        palette=["#D73027" if d < 0 else "#1A9850" for d in topk["delta_mean"]],
        ax=ax
    )

    ax.axvline(0, color="black", linewidth=1)

    ax.set_xlabel("Mean Δ Probability (Unhappy − Sad)")
    ax.set_ylabel("")
    ax.set_title("Aggregated intersectional shifts across Western models")

    ax.xaxis.set_major_formatter(lambda x, _: f"{x * 100:.1f}%")

    plt.tight_layout()
    plt.savefig("topk_intersection_shift_all_models.png", dpi=300)
    plt.show()


def vis_all():
    TOP_K = 20

    model_files = {
        "FLUX-Schnell": {
            "sad": "data/res/flux/sad_res.csv",
            "unhappy": "data/res/flux/unhappy_res.csv",
        },
        "Proteus": {
            "sad": "data/res/proteus/sad_res.csv",
            "unhappy": "data/res/proteus/unhappy_res.csv",
        },
        "SANA": {
            "sad": "data/res/sana/sad_res.csv",
            "unhappy": "data/res/sana/unhappy_res.csv",
        },
        "SD3": {
            "sad": "data/res/sd3/sad_res.csv",
            "unhappy": "data/res/sd3/unhappy_res.csv",
        },
        "Hunyuan": {
            "sad": "data/res/hunyuan/sad_res.csv",
            "unhappy": "data/res/hunyuan/unhappy_res.csv",
        },
        "Kolors": {
            "sad": "data/res/kolors/sad_res.csv",
            "unhappy": "data/res/kolors/unhappy_res.csv",
        },
        "Qwen": {
            "sad": "data/res/qwen/sad_res.csv",
            "unhappy": "data/res/qwen/unhappy_res.csv",
        },
        "Wan2.1": {
            "sad": "data/res/wan2.1/sad_res.csv",
            "unhappy": "data/res/wan2.1/unhappy_res.csv",
        }
    }

    # =====================================================
    # 1. Demographic processing (与你之前完全一致)
    # =====================================================

    def map_age(age):
        age = str(age)
        if age in ["0-2", "3-9"]:
            return "0-9"
        if age == "10-19":
            return "10-19"
        if age in ["20-29", "30-39"]:
            return "20-39"
        if age in ["40-49", "50-59"]:
            return "40-59"
        if age in ["60-69", "70+"]:
            return "60+"
        return None

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

    # =====================================================
    # 2. Utilities
    # =====================================================
    def load_and_process(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["face_name_align"].str.contains("face0")].copy()

        df["age"] = df["age"].apply(map_age)
        df["gender"] = df["gender"]
        df["race"] = df.apply(process_race, axis=1)

        return df

    def joint_distribution(df):
        joint = (
            df.groupby(["age", "gender", "race"])
            .size()
            .reset_index(name="count")
        )
        joint["prob"] = joint["count"] / joint["count"].sum()
        return joint[["age", "gender", "race", "prob"]]

    # =====================================================
    # 3. Compute aggregated Δ across models
    # =====================================================
    all_deltas = []

    for model, paths in model_files.items():
        df_sad = load_and_process(paths["sad"])
        df_unhappy = load_and_process(paths["unhappy"])

        p_sad = joint_distribution(df_sad)
        p_unhappy = joint_distribution(df_unhappy)

        merged = pd.merge(
            p_unhappy,
            p_sad,
            on=["age", "gender", "race"],
            how="outer",
            suffixes=("_unhappy", "_sad")
        ).fillna(0)

        merged["delta"] = merged["prob_unhappy"] - merged["prob_sad"]
        merged["model"] = model

        all_deltas.append(merged)

    all_deltas = pd.concat(all_deltas, ignore_index=True)

    # 👉 对模型取平均
    agg = (
        all_deltas
        .groupby(["age", "gender", "race"], as_index=False)
        .agg(delta_mean=("delta", "mean"))
    )

    # Label for plotting
    agg["group"] = (
            agg["age"] + " · " +
            agg["gender"] + " · " +
            agg["race"].str.capitalize()
    )

    # Select Top-K by absolute mean shift
    topk = (
        agg
        .reindex(agg["delta_mean"].abs().sort_values(ascending=False).index)
        .head(TOP_K)
        .sort_values("delta_mean")
    )

    # =====================================================
    # 4. Plot: Diverging bar chart (aggregated)
    # =====================================================
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        data=topk,
        x="delta_mean",
        y="group",
        palette=["#D73027" if d < 0 else "#1A9850" for d in topk["delta_mean"]],
        ax=ax
    )

    ax.axvline(0, color="black", linewidth=1)

    ax.set_xlabel("Mean Δ Probability (Unhappy − Sad)")
    ax.set_ylabel("")
    ax.set_title("Aggregated intersectional shifts across all models")

    ax.xaxis.set_major_formatter(lambda x, _: f"{x * 100:.1f}%")

    plt.tight_layout()
    plt.savefig("topk_intersection_shift_all_models.png", dpi=300)
    plt.show()


def vis_all2():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns

    TOP_K = 20

    model_files = {
        "FLUX-Schnell": {
            "sad":    "data/res/flux/sad_res.csv",
            "unhappy":"data/res/flux/unhappy_res.csv",
        },
        "Proteus": {
            "sad":    "data/res/proteus/sad_res.csv",
            "unhappy":"data/res/proteus/unhappy_res.csv",
        },
        "SANA": {
            "sad":    "data/res/sana/sad_res.csv",
            "unhappy":"data/res/sana/unhappy_res.csv",
        },
        "SD3": {
            "sad":    "data/res/sd3/sad_res.csv",
            "unhappy":"data/res/sd3/unhappy_res.csv",
        },
        "Hunyuan": {
            "sad":    "data/res/hunyuan/sad_res.csv",
            "unhappy":"data/res/hunyuan/unhappy_res.csv",
        },
        "Kolors": {
            "sad":    "data/res/kolors/sad_res.csv",
            "unhappy":"data/res/kolors/unhappy_res.csv",
        },
        "Qwen": {
            "sad":    "data/res/qwen/sad_res.csv",
            "unhappy":"data/res/qwen/unhappy_res.csv",
        },
        "Wan2.1": {
            "sad":    "data/res/wan2.1/sad_res.csv",
            "unhappy":"data/res/wan2.1/unhappy_res.csv",
        }
    }

    # =====================================================
    # 1. Demographic processing
    # =====================================================
    def map_age(age):
        age = str(age)
        if age in ["0-2", "3-9"]:       return "0-9"
        if age == "10-19":              return "10-19"
        if age in ["20-29", "30-39"]:   return "20-39"
        if age in ["40-49", "50-59"]:   return "40-59"
        if age in ["60-69", "70+"]:     return "60+"
        return None

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
        if race in ["Indian", "Latino_Hispanic"]: return "others"
        if race in ["East Asian", "Southeast Asian"]: return "asian"
        return race.lower()

    # =====================================================
    # 2. Utilities
    # =====================================================
    def load_and_process(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["face_name_align"].str.contains("face0")].copy()
        df["age"]    = df["age"].apply(map_age)
        df["gender"] = df["gender"]
        df["race"]   = df.apply(process_race, axis=1)
        return df

    def joint_distribution(df):
        joint = (
            df.groupby(["age", "gender", "race"])
            .size()
            .reset_index(name="count")
        )
        joint["prob"] = joint["count"] / joint["count"].sum()
        return joint[["age", "gender", "race", "prob"]]

    # =====================================================
    # 3. Compute aggregated Δ across models
    # =====================================================
    all_deltas = []

    for model, paths in model_files.items():
        df_sad     = load_and_process(paths["sad"])
        df_unhappy = load_and_process(paths["unhappy"])
        p_sad      = joint_distribution(df_sad)
        p_unhappy  = joint_distribution(df_unhappy)

        merged = pd.merge(
            p_unhappy, p_sad,
            on=["age", "gender", "race"],
            how="outer",
            suffixes=("_unhappy", "_sad")
        ).fillna(0)

        merged["delta"] = merged["prob_unhappy"] - merged["prob_sad"]
        merged["model"] = model
        all_deltas.append(merged)

    all_deltas = pd.concat(all_deltas, ignore_index=True)

    agg = (
        all_deltas
        .groupby(["age", "gender", "race"], as_index=False)
        .agg(delta_mean=("delta", "mean"))
    )

    agg["group"] = (
        agg["age"] + " · " +
        agg["gender"] + " · " +
        agg["race"].str.capitalize()
    )

    topk = (
        agg
        .reindex(agg["delta_mean"].abs().sort_values(ascending=False).index)
        .head(TOP_K)
        .sort_values("delta_mean")
    )

    # =====================================================
    # 4. Color: intensity encodes magnitude
    # =====================================================
    red_cmap   = mcolors.LinearSegmentedColormap.from_list("reds",   ["#FCBBA1", "#67000D"])
    green_cmap = mcolors.LinearSegmentedColormap.from_list("greens", ["#C7E9C0", "#00441B"])

    vals     = topk["delta_mean"].values
    abs_vals = np.abs(vals)
    max_abs  = abs_vals.max()

    bar_colors = []
    for v in vals:
        intensity = abs(v) / max_abs          # 0 → light, 1 → dark
        if v < 0:
            bar_colors.append(red_cmap(0.3 + 0.7 * intensity))
        else:
            bar_colors.append(green_cmap(0.3 + 0.7 * intensity))

    # =====================================================
    # 5. Plot
    # =====================================================
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)

    fig, ax = plt.subplots(figsize=(9, 7))

    # 轻薄横向网格
    ax.xaxis.grid(True, linewidth=0.4, color="#d0d0d0", alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    bars = ax.barh(
        topk["group"],
        topk["delta_mean"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.4,
        height=0.65,
        zorder=3,
    )

    # 零线
    ax.axvline(0, color="#333333", linewidth=0.8, zorder=4)

    # 数值标注
    for bar, v in zip(bars, vals):
        x_end  = bar.get_width()
        offset = 0.0003 if v >= 0 else -0.0003
        ha     = "left" if v >= 0 else "right"
        ax.text(
            x_end + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{v * 100:+.2f}%",
            va="center", ha=ha,
            fontsize=10, color="#333333"
        )

    ax.set_xlabel("Mean Δ Probability (Unhappy − Sad)", fontsize=13)
    ax.set_ylabel("", fontsize=13)
    ax.set_title("Aggregated intersectional shifts across all models", fontsize=14, pad=10)
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", labelsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))

    # x 轴范围留出标注空间
    x_margin = max_abs * 0.18
    ax.set_xlim(-max_abs - x_margin, max_abs + x_margin)

    sns.despine(ax=ax, left=True, bottom=False)

    plt.tight_layout()
    plt.savefig("topk_intersection_shift_all_models.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    # western_race()
    # chinese_race()
    # western_age()
    # chinese_age()
    #western_gender()
    #chinese_gender()
    #js_computation()
    # vis_western()
    # vis_chinese()
    vis_all2()