
import json, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from matplotlib.colors import LinearSegmentedColormap

red_white_green = LinearSegmentedColormap.from_list(
    "red_white_green",
    [
        (0.0, "#6D2E46"),   # 负极值：红（与你 bar chart 一致）
        (0.45, "#f7f7f7"),  # 0 附近：浅灰
        (0.55, "#f7f7f7"),
        (1.0, "#1e8449"),   # 正极值：绿（与你 bar chart 一致）
    ]
)

def flux_age_heatmap():
    base_path = "/content/bias/age"
    model = "flux"

    with open(os.path.join(base_path, f"{model}_age_counts.json"), "r") as f:
        data = json.load(f)

    age_bins = ["0-9", "10-19", "20-39", "40-59", "60+"]
    emotions = ["surprised", "happy", "sad", "disgusted", "angry", "fearful"]

    # 中性分布
    neutral = data["person"]
    total_neutral = sum(neutral.values())
    P_neutral = np.array([neutral[b] / total_neutral for b in age_bins])

    # 计算 ΔP
    delta_matrix = []
    for emo in emotions:
        counts = data[emo]
        total = sum(counts.values())
        P_emo = np.array([counts[b] / total for b in age_bins])
        delta = P_emo - P_neutral  # 带符号差值
        delta_matrix.append(delta)

    delta_matrix = np.vstack(delta_matrix)

    plt.figure(figsize=(7, 4))
    sns.heatmap(
        delta_matrix,
        xticklabels=age_bins,
        yticklabels=emotions,
        center=0.0, cmap="RdBu_r", annot=True, fmt=".2f"
    )
    plt.title(f"{model}: Emotion-conditioned shift relative to neutral (ΔP)")
    plt.xlabel("Age group")
    plt.ylabel("Emotion")
    plt.tight_layout()
    plt.show()




def w_c_age_heatmap():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # ----------------------------
    # Define model groups
    # ----------------------------
    western_models = ["flux", "proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]
    all_models = western_models + chinese_models

    # ----------------------------
    # Paths and metadata
    # ----------------------------
    base_path = "/Users/wmt/projects/bias/write_into_json/age"
    age_bins = ["0-9", "10-19", "20-39", "40-59", "60+"]

    # *** REQUIRED EMOTION ORDER (for indexing JSON) ***
    emotions = ["happy", "surprised", "sad", "fearful", "angry", "disgusted"]

    # --- display names (ONLY for visualization) ---
    emotion_display = {
        "surprised": "Sur.",
        "sad": "Sad",
        "happy": "Hap.",
        "angry": "Ang.",
        "disgusted": "Dis.",
        "fearful": "Fear",
    }
    emotion_labels = [emotion_display[e] for e in emotions]

    # ----------------------------
    # Function to compute ΔP
    # ----------------------------
    def compute_delta_matrix(model):
        with open(os.path.join(base_path, f"{model}_age.json"), "r") as f:
            data = json.load(f)

        neutral = data["neutral"]
        P_neutral = np.array([neutral[b] / sum(neutral.values()) for b in age_bins])

        deltas = []
        for emo in emotions:
            emo_counts = data[emo]
            P_emo = np.array([emo_counts[b] / sum(emo_counts.values()) for b in age_bins])
            deltas.append(P_emo - P_neutral)

        return np.array(deltas)

    # ----------------------------
    # Compute group averages
    # ----------------------------
    western_delta = np.mean([compute_delta_matrix(m) for m in western_models], axis=0)
    chinese_delta = np.mean([compute_delta_matrix(m) for m in chinese_models], axis=0)
    all_delta = np.mean([compute_delta_matrix(m) for m in all_models], axis=0)

    # ----------------------------
    # Heatmap drawing helper
    # ----------------------------
    def draw_heatmap(delta_matrix, title, filename):
        plt.figure(figsize=(7, 5))
        ax = sns.heatmap(
            delta_matrix,
            xticklabels=age_bins,
            yticklabels=emotion_labels,  # 👈 使用“修改后的 emotion 名称”
            center=0.0,
            cmap=red_white_green,
            annot=True,
            fmt=".2f",
            vmin=-0.4,
            vmax=0.4,
            annot_kws={"size": 20},
            cbar=True
        )

        # ---------- 标题 & 轴 ----------
        plt.title(title, fontsize=16)
        plt.ylabel("Emotion", fontsize=20)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # ---------- Colorbar 字体放大 ----------
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)  # 刻度字体
        # cbar.set_label("Mean ΔP", fontsize=20)  # colorbar 标题

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    # ----------------------------
    # Save all heatmaps
    # ----------------------------
    draw_heatmap(
        western_delta,
        "Western Models (Mean ΔP on Age Distribution)",
        "western_age_delta.png"
    )
    draw_heatmap(
        chinese_delta,
        "Chinese Models (Mean ΔP on Age Distribution)",
        "chinese_age_delta.png"
    )
    draw_heatmap(
        all_delta,
        "All Models (Mean ΔP on Age Distribution)",
        "allmodels_age_delta.png"
    )

    print("Saved: western_age_delta.png, chinese_age_delta.png, allmodels_age_delta.png")


def w_c_age_heatmap2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns

    sns.set_style("white")

    # ----------------------------
    # 1. Model groups
    # ----------------------------
    western_models = ["flux", "proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]
    all_models = western_models + chinese_models

    # ----------------------------
    # 2. Paths and metadata
    # ----------------------------
    base_path = "/Users/wmt/projects/bias/write_into_json/age"
    age_bins = ["0-9", "10-19", "20-39", "40-59", "60+"]
    emotions = ["happy", "surprised", "sad", "fearful", "angry", "disgusted"]

    emotion_display = {
        "surprised": "Surprise",
        "sad": "Sadness",
        "happy": "Happiness",
        "angry": "Anger",
        "disgusted": "Disgust",
        "fearful": "Fear",
    }

    # ----------------------------
    # 3. Custom red-white-green colormap
    # ----------------------------
    red_white_green = mcolors.LinearSegmentedColormap.from_list(
        "rwg", ["#D73027", "#FFFFFF", "#1A9850"], N=256
    )

    # ----------------------------
    # 4. Compute ΔP matrix
    # ----------------------------
    def compute_delta_matrix(model):
        with open(os.path.join(base_path, f"{model}_age.json"), "r") as f:
            data = json.load(f)

        neutral = data["neutral"]
        P_neutral = np.array([neutral[b] / sum(neutral.values()) for b in age_bins])

        deltas = []
        for emo in emotions:
            emo_counts = data[emo]
            P_emo = np.array([emo_counts[b] / sum(emo_counts.values()) for b in age_bins])
            deltas.append(P_emo - P_neutral)

        return np.array(deltas)  # shape: (n_emotions, n_age_bins)

    # ----------------------------
    # 5. Group averages
    # ----------------------------
    western_delta = np.mean([compute_delta_matrix(m) for m in western_models], axis=0)
    chinese_delta = np.mean([compute_delta_matrix(m) for m in chinese_models], axis=0)

    # ----------------------------
    # 6. Sort emotions by mean |ΔP| across both groups
    # ----------------------------
    combined = np.mean([western_delta, chinese_delta], axis=0)
    mean_abs = np.mean(np.abs(combined), axis=1)
    sort_idx = np.argsort(mean_abs)[::-1]

    sorted_emotions = [emotions[i] for i in sort_idx]
    sorted_labels = [emotion_display[e] for e in sorted_emotions]

    western_delta = western_delta[sort_idx]
    chinese_delta = chinese_delta[sort_idx]

    # ----------------------------
    # 8. Plot: 2 rows × 1 col (vertical), no All Models
    # ----------------------------
    vmin, vmax = -0.4, 0.4

    fig, axes = plt.subplots(
        2, 1,
        figsize=(7, 10),
        gridspec_kw={"hspace": 0.12}
    )

    panels = [
        (western_delta, "Western Models"),
        (chinese_delta, "Chinese Models"),
    ]

    for idx, (ax, (delta, title)) in enumerate(zip(axes, panels)):
        sns.heatmap(
            delta,
            ax=ax,
            xticklabels=age_bins,
            yticklabels=sorted_labels,
            center=0.0,
            cmap=red_white_green,
            annot=True,
            fmt="+.2f",
            vmin=vmin,
            vmax=vmax,
            annot_kws={"size": 20, "weight": "normal"},
            cbar=False,
            linewidths=0.4,
            linecolor="#e0e0e0",
        )

        ax.set_title(title, fontsize=18, pad=10)
        ax.tick_params(axis="x", labelsize=18, rotation=0)
        ax.set_ylabel("Emotion", fontsize=20)
        ax.tick_params(axis="y", labelsize=18, rotation=0)

        # x 轴标签只在最下面一行显示
        if idx == 0:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    # ---------- Shared colorbar ----------
    sm = plt.cm.ScalarMappable(
        cmap=red_white_green,
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Mean ΔP", fontsize=16, labelpad=8)

    plt.savefig("age_delta_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved: age_delta_heatmap.png")


def w_c_gender_heatmap():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # ----------------------------
    # Define model groups
    # ----------------------------
    western_models = ["flux", "proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]
    all_models = western_models + chinese_models

    # ----------------------------
    # Paths and metadata
    # ----------------------------
    base_path = "/Users/wmt/projects/bias/write_into_json/gender"
    gender_bins = ["Male", "Female"]

    # *** REQUIRED EMOTION ORDER (for indexing JSON) ***
    emotions = ["happy", "surprised", "sad", "fearful", "angry", "disgusted"]

    # --- display names (ONLY for visualization) ---
    emotion_display = {
        "surprised": "Sur.",
        "sad": "Sad",
        "happy": "Hap.",
        "angry": "Ang.",
        "disgusted": "Dis.",
        "fearful": "Fear",
    }

    emotion_labels = [emotion_display[e] for e in emotions]

    # ----------------------------
    # Function to compute ΔP
    # ----------------------------
    def compute_delta_matrix(model):
        with open(os.path.join(base_path, f"{model}_gender.json"), "r") as f:
            data = json.load(f)

        neutral = data["neutral"]
        total_neutral = sum(neutral.values())
        P_neutral = np.array([neutral[b] / total_neutral for b in gender_bins])

        deltas = []
        for emo in emotions:
            emo_counts = data[emo]
            total_emo = sum(emo_counts.values())
            P_emo = np.array([emo_counts[b] / total_emo for b in gender_bins])
            deltas.append(P_emo - P_neutral)

        return np.array(deltas)

    # ----------------------------
    # Compute group averages
    # ----------------------------
    western_delta = np.mean([compute_delta_matrix(m) for m in western_models], axis=0)
    chinese_delta = np.mean([compute_delta_matrix(m) for m in chinese_models], axis=0)
    all_delta = np.mean([compute_delta_matrix(m) for m in all_models], axis=0)

    # ----------------------------
    # Heatmap drawing helper
    # ----------------------------
    def draw_heatmap(delta_matrix, title, filename):
        plt.figure(figsize=(7, 5))
        ax = sns.heatmap(
            delta_matrix,
            xticklabels=gender_bins,
            yticklabels=emotion_labels,
            center=0.0,
            cmap=red_white_green,
            annot=True,
            fmt=".2f",
            vmin=-0.4,
            vmax=0.4,
            annot_kws={"size": 20},
            cbar=True
        )

        # ---------- 标题 & 轴 ----------
        plt.title(title, fontsize=16)
        plt.ylabel("Emotion", fontsize=20)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # ---------- Colorbar 字体 ----------
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    # ----------------------------
    # Save all heatmaps
    # ----------------------------
    draw_heatmap(
        western_delta,
        "Western Models (Mean ΔP on Gender Distribution)",
        "western_gender_delta.png"
    )

    draw_heatmap(
        chinese_delta,
        "Chinese Models (Mean ΔP on Gender Distribution)",
        "chinese_gender_delta.png"
    )

    draw_heatmap(
        all_delta,
        "All Models (Mean ΔP on Gender Distribution)",
        "allmodels_gender_delta.png"
    )

    print("Saved: western_gender_delta.png, chinese_gender_delta.png, allmodels_gender_delta.png")

def w_c_gender_heatmap2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns

    sns.set_style("white")

    # ----------------------------
    # 1. Model groups
    # ----------------------------
    western_models = ["flux", "proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]

    # ----------------------------
    # 2. Paths and metadata
    # ----------------------------
    base_path = "/Users/wmt/projects/bias/write_into_json/gender"
    gender_bins = ["Male", "Female"]
    emotions = ["happy", "surprised", "sad", "fearful", "angry", "disgusted"]

    emotion_display = {
        "surprised": "Surprise",
        "sad": "Sadness",
        "happy": "Happiness",
        "angry": "Anger",
        "disgusted": "Disgust",
        "fearful": "Fear",
    }

    # ----------------------------
    # 3. Custom red-white-green colormap
    # ----------------------------
    red_white_green = mcolors.LinearSegmentedColormap.from_list(
        "rwg", ["#D73027", "#FFFFFF", "#1A9850"], N=256
    )

    # ----------------------------
    # 4. Compute ΔP matrix
    # ----------------------------
    def compute_delta_matrix(model):
        with open(os.path.join(base_path, f"{model}_gender.json"), "r") as f:
            data = json.load(f)

        neutral = data["neutral"]
        P_neutral = np.array([neutral[b] / sum(neutral.values()) for b in gender_bins])

        deltas = []
        for emo in emotions:
            emo_counts = data[emo]
            P_emo = np.array([emo_counts[b] / sum(emo_counts.values()) for b in gender_bins])
            deltas.append(P_emo - P_neutral)

        return np.array(deltas)  # shape: (n_emotions, n_gender_bins)

    # ----------------------------
    # 5. Group averages
    # ----------------------------
    western_delta = np.mean([compute_delta_matrix(m) for m in western_models], axis=0)
    chinese_delta = np.mean([compute_delta_matrix(m) for m in chinese_models], axis=0)

    # ----------------------------
    # 6. Sort emotions by mean |ΔP| across both groups
    # ----------------------------
    combined = np.mean([western_delta, chinese_delta], axis=0)
    mean_abs = np.mean(np.abs(combined), axis=1)
    sort_idx = np.argsort(mean_abs)[::-1]

    sorted_emotions = [emotions[i] for i in sort_idx]
    sorted_labels = [emotion_display[e] for e in sorted_emotions]

    western_delta = western_delta[sort_idx]
    chinese_delta = chinese_delta[sort_idx]

    # ----------------------------
    # 7. Plot: 2 rows × 1 col (vertical), shared colorbar
    # ----------------------------
    vmin, vmax = -0.4, 0.4

    fig, axes = plt.subplots(
        2, 1,
        figsize=(5, 10),
        gridspec_kw={"hspace": 0.12}
    )

    panels = [
        (western_delta, "Western Models"),
        (chinese_delta, "Chinese Models"),
    ]

    for idx, (ax, (delta, title)) in enumerate(zip(axes, panels)):
        sns.heatmap(
            delta,
            ax=ax,
            xticklabels=gender_bins,
            yticklabels=sorted_labels,
            center=0.0,
            cmap=red_white_green,
            annot=True,
            fmt="+.2f",
            vmin=vmin,
            vmax=vmax,
            annot_kws={"size": 20, "weight": "normal"},
            cbar=False,
            linewidths=0.4,
            linecolor="#e0e0e0",
        )

        ax.set_title(title, fontsize=18, pad=10)
        ax.tick_params(axis="x", labelsize=18, rotation=0)
        ax.set_ylabel("Emotion", fontsize=20)
        ax.tick_params(axis="y", labelsize=18, rotation=0)

        # x 轴标签只在最下面一行显示
        if idx == 0:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    # ---------- Shared colorbar ----------
    sm = plt.cm.ScalarMappable(
        cmap=red_white_green,
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Mean ΔP", fontsize=16, labelpad=8)

    plt.savefig("gender_delta_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved: gender_delta_heatmap.png")

def w_c_race_heatmap():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # ----------------------------
    # Define model groups
    # ----------------------------
    western_models = ["flux", "proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]
    all_models = western_models + chinese_models

    # ----------------------------
    # Paths and metadata
    # ----------------------------
    base_path = "/Users/wmt/projects/bias/write_into_json/race"
    race_bins = ["White", "Black", "Asian", "Others"]

    # *** REQUIRED EMOTION ORDER (for indexing JSON) ***
    emotions = ["happy", "surprised", "sad", "fearful", "angry", "disgusted"]

    # --- display names (ONLY for visualization) ---
    emotion_display = {
        "surprised": "Sur.",
        "sad": "Sad",
        "happy": "Hap.",
        "angry": "Ang.",
        "disgusted": "Dis.",
        "fearful": "Fear",
    }

    emotion_labels = [emotion_display[e] for e in emotions]

    # ----------------------------
    # Function to compute ΔP matrix for 1 model
    # ----------------------------
    def compute_delta_matrix(model):
        with open(os.path.join(base_path, f"{model}_race.json"), "r") as f:
            data = json.load(f)

        neutral = data["neutral"]
        total_neutral = sum(neutral.values())
        P_neutral = np.array([neutral[b] / total_neutral for b in race_bins])

        deltas = []
        for emo in emotions:
            emo_counts = data[emo]
            total_emo = sum(emo_counts.values())
            P_emo = np.array([emo_counts[b] / total_emo for b in race_bins])
            deltas.append(P_emo - P_neutral)

        return np.array(deltas)

    # ----------------------------
    # Compute group means
    # ----------------------------
    western_delta = np.mean([compute_delta_matrix(m) for m in western_models], axis=0)
    chinese_delta = np.mean([compute_delta_matrix(m) for m in chinese_models], axis=0)
    all_delta = np.mean([compute_delta_matrix(m) for m in all_models], axis=0)

    # ----------------------------
    # Helper to draw & save heatmaps
    # ----------------------------
    def draw_heatmap(delta_matrix, title, filename):
        plt.figure(figsize=(7, 5))
        ax = sns.heatmap(
            delta_matrix,
            xticklabels=race_bins,
            yticklabels=emotion_labels,
            center=0.0,
            cmap=red_white_green,
            annot=True,
            fmt=".2f",
            vmin=-0.4,
            vmax=0.4,
            annot_kws={"size": 20},
            cbar=True
        )

        # ---------- 标题 & 轴 ----------
        plt.title(title, fontsize=16)
        plt.ylabel("Emotion", fontsize=20)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # ---------- Colorbar 字体 ----------
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    # ----------------------------
    # Draw & save all heatmaps
    # ----------------------------
    draw_heatmap(
        western_delta,
        "Western Models (Mean ΔP on Race Distribution)",
        "western_race_delta.png"
    )

    draw_heatmap(
        chinese_delta,
        "Chinese Models (Mean ΔP on Race Distribution)",
        "chinese_race_delta.png"
    )

    draw_heatmap(
        all_delta,
        "All Models (Mean ΔP on Race Distribution)",
        "allmodels_race_delta.png"
    )

    print("Saved: western_race_delta.png, chinese_race_delta.png, allmodels_race_delta.png")

def w_c_race_heatmap2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns

    sns.set_style("white")

    # ----------------------------
    # 1. Model groups
    # ----------------------------
    western_models = ["flux", "proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]

    # ----------------------------
    # 2. Paths and metadata
    # ----------------------------
    base_path = "/Users/wmt/projects/bias/write_into_json/race"
    race_bins = ["White", "Black", "Asian", "Others"]
    emotions = ["happy", "surprised", "sad", "fearful", "angry", "disgusted"]

    emotion_display = {
        "surprised": "Surprise",
        "sad": "Sadness",
        "happy": "Happiness",
        "angry": "Anger",
        "disgusted": "Disgust",
        "fearful": "Fear",
    }

    # ----------------------------
    # 3. Custom red-white-green colormap
    # ----------------------------
    red_white_green = mcolors.LinearSegmentedColormap.from_list(
        "rwg", ["#D73027", "#FFFFFF", "#1A9850"], N=256
    )

    # ----------------------------
    # 4. Compute ΔP matrix
    # ----------------------------
    def compute_delta_matrix(model):
        with open(os.path.join(base_path, f"{model}_race.json"), "r") as f:
            data = json.load(f)

        neutral = data["neutral"]
        P_neutral = np.array([neutral[b] / sum(neutral.values()) for b in race_bins])

        deltas = []
        for emo in emotions:
            emo_counts = data[emo]
            P_emo = np.array([emo_counts[b] / sum(emo_counts.values()) for b in race_bins])
            deltas.append(P_emo - P_neutral)

        return np.array(deltas)  # shape: (n_emotions, n_race_bins)

    # ----------------------------
    # 5. Group averages
    # ----------------------------
    western_delta = np.mean([compute_delta_matrix(m) for m in western_models], axis=0)
    chinese_delta = np.mean([compute_delta_matrix(m) for m in chinese_models], axis=0)

    # ----------------------------
    # 6. Sort emotions by mean |ΔP| across both groups
    # ----------------------------
    combined = np.mean([western_delta, chinese_delta], axis=0)
    mean_abs = np.mean(np.abs(combined), axis=1)
    sort_idx = np.argsort(mean_abs)[::-1]

    sorted_emotions = [emotions[i] for i in sort_idx]
    sorted_labels = [emotion_display[e] for e in sorted_emotions]

    western_delta = western_delta[sort_idx]
    chinese_delta = chinese_delta[sort_idx]

    # ----------------------------
    # 7. Plot: 2 rows × 1 col (vertical), shared colorbar
    # ----------------------------
    vmin, vmax = -0.4, 0.4

    fig, axes = plt.subplots(
        2, 1,
        figsize=(7, 10),
        gridspec_kw={"hspace": 0.12}
    )

    panels = [
        (western_delta, "Western Models"),
        (chinese_delta, "Chinese Models"),
    ]

    for idx, (ax, (delta, title)) in enumerate(zip(axes, panels)):
        sns.heatmap(
            delta,
            ax=ax,
            xticklabels=race_bins,
            yticklabels=sorted_labels,
            center=0.0,
            cmap=red_white_green,
            annot=True,
            fmt="+.2f",
            vmin=vmin,
            vmax=vmax,
            annot_kws={"size": 20, "weight": "normal"},
            cbar=False,
            linewidths=0.4,
            linecolor="#e0e0e0",
        )

        ax.set_title(title, fontsize=18, pad=10)
        ax.tick_params(axis="x", labelsize=18, rotation=0)
        ax.set_ylabel("Emotion", fontsize=20)
        ax.tick_params(axis="y", labelsize=18, rotation=0)

        # x 轴标签只在最下面一行显示
        if idx == 0:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    # ---------- Shared colorbar ----------
    sm = plt.cm.ScalarMappable(
        cmap=red_white_green,
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Mean ΔP", fontsize=16, labelpad=8)

    plt.savefig("race_delta_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved: race_delta_heatmap.png")

def w_c_attractive_heatmap():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    # 数值 → 语义映射
    attr_map = {
        "low": "0",
        "medium": "1",
        "high": "2"
    }
    # ----------------------------
    # Define model groups
    # ----------------------------
    western_models = ["flux", "proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]
    all_models = western_models + chinese_models

    # ----------------------------
    # Paths and metadata
    # ----------------------------
    base_path = "/Users/wmt/projects/bias/write_into_json/attractiveness"
    attr_bins = ["low", "medium", "high"]

    # *** REQUIRED EMOTION ORDER (for indexing JSON) ***
    emotions = ["happy", "surprised", "sad", "fearful", "angry", "disgusted"]

    # --- display names (ONLY for visualization) ---
    emotion_display = {
        "surprised": "Sur.",
        "sad": "Sad",
        "happy": "Hap.",
        "angry": "Ang.",
        "disgusted": "Dis.",
        "fearful": "Fear",
    }

    emotion_labels = [emotion_display[e] for e in emotions]

    # ----------------------------
    # Function to compute ΔP for 1 model
    # ----------------------------
    def compute_delta_matrix(model):
        with open(os.path.join(base_path, f"{model}_attractiveness.json"), "r") as f:
            data = json.load(f)

        # ---------- neutral ----------
        neutral = data["neutral"]
        total_neutral = sum(neutral.values())

        if total_neutral == 0:
            P_neutral = np.zeros(len(attr_bins))
        else:
            P_neutral = np.array([
                neutral.get(attr_map[b], 0) / total_neutral
                for b in attr_bins
            ])

        # ---------- emotions ----------
        deltas = []
        for emo in emotions:
            emo_counts = data[emo]
            total_emo = sum(emo_counts.values())

            if total_emo == 0:
                P_emo = np.zeros(len(attr_bins))
            else:
                P_emo = np.array([
                    emo_counts.get(attr_map[b], 0) / total_emo
                    for b in attr_bins
                ])

            deltas.append(P_emo - P_neutral)

        return np.array(deltas)

    # ----------------------------
    # Compute ΔP for each group
    # ----------------------------
    western_delta = np.mean([compute_delta_matrix(m) for m in western_models], axis=0)
    chinese_delta = np.mean([compute_delta_matrix(m) for m in chinese_models], axis=0)
    all_delta = np.mean([compute_delta_matrix(m) for m in all_models], axis=0)

    # ----------------------------
    # Helper to draw & save heatmaps
    # ----------------------------
    def draw_heatmap(delta_matrix, title, filename):
        plt.figure(figsize=(7, 5))
        ax = sns.heatmap(
            delta_matrix,
            xticklabels=attr_bins,
            yticklabels=emotion_labels,
            center=0.0,
            cmap=red_white_green,
            annot=True,
            fmt=".2f",
            vmin=-0.4,
            vmax=0.4,
            annot_kws={"size": 20},
            cbar=True
        )

        # ---------- 标题 & 轴 ----------
        plt.title(title, fontsize=16)
        plt.ylabel("Emotion", fontsize=20)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # ---------- Colorbar 字体 ----------
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    # ----------------------------
    # Draw & save all three heatmaps
    # ----------------------------
    draw_heatmap(
        western_delta,
        "Western Models (Mean ΔP on Attractiveness Distribution)",
        "western_attractiveness_delta.png"
    )

    draw_heatmap(
        chinese_delta,
        "Chinese Models (Mean ΔP on Attractiveness Distribution)",
        "chinese_attractiveness_delta.png"
    )

    draw_heatmap(
        all_delta,
        "All Models (Mean ΔP on Attractiveness Distribution)",
        "allmodels_attractiveness_delta.png"
    )

    print(
        "Saved: western_attractiveness_delta.png, "
        "chinese_attractiveness_delta.png, "
        "allmodels_attractiveness_delta.png"
    )

def w_c_attractive_heatmap2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns

    sns.set_style("white")

    # ----------------------------
    # 1. Model groups
    # ----------------------------
    western_models = ["flux", "proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "wan2.1"]

    # ----------------------------
    # 2. Paths and metadata
    # ----------------------------
    base_path = "/Users/wmt/projects/bias/write_into_json/attractiveness"
    attr_bins = ["low", "medium", "high"]
    attr_map = {"low": "0", "medium": "1", "high": "2"}
    emotions = ["happy", "surprised", "sad", "fearful", "angry", "disgusted"]

    emotion_display = {
        "surprised": "Surprise",
        "sad": "Sadness",
        "happy": "Happiness",
        "angry": "Anger",
        "disgusted": "Disgust",
        "fearful": "Fear",
    }

    # ----------------------------
    # 3. Custom red-white-green colormap
    # ----------------------------
    red_white_green = mcolors.LinearSegmentedColormap.from_list(
        "rwg", ["#D73027", "#FFFFFF", "#1A9850"], N=256
    )

    # ----------------------------
    # 4. Compute ΔP matrix
    # ----------------------------
    def compute_delta_matrix(model):
        with open(os.path.join(base_path, f"{model}_attractiveness.json"), "r") as f:
            data = json.load(f)

        neutral = data["neutral"]
        total_neutral = sum(neutral.values())
        P_neutral = (
            np.zeros(len(attr_bins)) if total_neutral == 0
            else np.array([neutral.get(attr_map[b], 0) / total_neutral for b in attr_bins])
        )

        deltas = []
        for emo in emotions:
            emo_counts = data[emo]
            total_emo = sum(emo_counts.values())
            P_emo = (
                np.zeros(len(attr_bins)) if total_emo == 0
                else np.array([emo_counts.get(attr_map[b], 0) / total_emo for b in attr_bins])
            )
            deltas.append(P_emo - P_neutral)

        return np.array(deltas)  # shape: (n_emotions, n_attr_bins)

    # ----------------------------
    # 5. Group averages
    # ----------------------------
    western_delta = np.mean([compute_delta_matrix(m) for m in western_models], axis=0)
    chinese_delta = np.mean([compute_delta_matrix(m) for m in chinese_models], axis=0)

    # ----------------------------
    # 6. Sort emotions by mean |ΔP| across both groups
    # ----------------------------
    combined = np.mean([western_delta, chinese_delta], axis=0)
    mean_abs = np.mean(np.abs(combined), axis=1)
    sort_idx = np.argsort(mean_abs)[::-1]

    sorted_emotions = [emotions[i] for i in sort_idx]
    sorted_labels = [emotion_display[e] for e in sorted_emotions]

    western_delta = western_delta[sort_idx]
    chinese_delta = chinese_delta[sort_idx]

    # ----------------------------
    # 7. Plot: 2 rows × 1 col (vertical), shared colorbar
    # ----------------------------
    vmin, vmax = -0.4, 0.4

    fig, axes = plt.subplots(
        2, 1,
        figsize=(6, 10),
        gridspec_kw={"hspace": 0.12}
    )

    panels = [
        (western_delta, "Western Models"),
        (chinese_delta, "Chinese Models"),
    ]

    for idx, (ax, (delta, title)) in enumerate(zip(axes, panels)):
        sns.heatmap(
            delta,
            ax=ax,
            xticklabels=attr_bins,
            yticklabels=sorted_labels,
            center=0.0,
            cmap=red_white_green,
            annot=True,
            fmt="+.2f",
            vmin=vmin,
            vmax=vmax,
            annot_kws={"size": 20, "weight": "normal"},
            cbar=False,
            linewidths=0.4,
            linecolor="#e0e0e0",
        )

        ax.set_title(title, fontsize=18, pad=10)
        ax.tick_params(axis="x", labelsize=18, rotation=0)
        ax.set_ylabel("Emotion", fontsize=20)
        ax.tick_params(axis="y", labelsize=18, rotation=0)

        # x 轴标签只在最下面一行显示
        if idx == 0:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    # ---------- Shared colorbar ----------
    sm = plt.cm.ScalarMappable(
        cmap=red_white_green,
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Mean ΔP", fontsize=16, labelpad=8)

    plt.savefig("attractiveness_delta_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved: attractiveness_delta_heatmap.png")


if __name__ == '__main__':
    w_c_age_heatmap2()
    w_c_gender_heatmap2()
    w_c_race_heatmap2()
    w_c_attractive_heatmap2()