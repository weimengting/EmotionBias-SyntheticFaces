import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os



def age_influence():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # ---------------------------------------------------
    # 1. Models, groups, colors, legend names
    # ---------------------------------------------------
    western_models = ["flux", "Proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "Wan2.1"]
    model_names = western_models + chinese_models

    model_colors = {
        "flux": "#08306B",
        "Proteus": "#2171B5",
        "sd3": "#C6DBEF",
        "sana": "#6BAED6",
        "hunyuan": "#7F2704",
        "qwen": "#E6550D",
        "kolors": "#A63603",
        "Wan2.1": "#FDBE85",
    }

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

    base_path = "/Users/wmt/projects/bias/data/jsons/age"

    # ---------------------------------------------------
    # 2. KL divergence
    # ---------------------------------------------------
    def kl_divergence(p, q, eps=1e-12):
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return np.sum(p * np.log(p / q))

    age_keys = ["0-9", "10-19", "20-39", "40-59", "60+"]

    emotions = ["surprised", "happy", "sad", "disgusted", "angry", "fearful"]

    # display name mapping
    emotion_display = {
        "surprised": "surprise",
        "happy": "happiness",
        "sad": "sadness",
        "disgusted": "disgust",
        "angry": "anger",
        "fearful": "fear",
    }

    # ---------------------------------------------------
    # 3. Compute KL(emotion || neutral)
    # ---------------------------------------------------
    kl_results = {m: [] for m in model_names}

    for model in model_names:
        with open(os.path.join(base_path, f"{model}_age_counts.json"), "r") as f:
            data = json.load(f)

        neutral_counts = data["person"]
        total_neutral = sum(neutral_counts.values())
        P_neutral = np.array([neutral_counts[k] / total_neutral for k in age_keys])

        for emo in emotions:
            emo_counts = data[emo]
            total_emo = sum(emo_counts.values())
            P_emo = np.array([emo_counts[k] / total_emo for k in age_keys])

            KL = kl_divergence(P_emo, P_neutral)
            kl_results[model].append(KL)

    # ---------------------------------------------------
    # 4. Sort emotions by mean KL
    # ---------------------------------------------------
    emotion_means = {
        emo: np.mean([kl_results[m][i] for m in model_names])
        for i, emo in enumerate(emotions)
    }

    sorted_emotions = sorted(emotions, key=lambda e: emotion_means[e])
    sorted_x = np.arange(len(sorted_emotions))

    sorted_kl = {
        m: [kl_results[m][emotions.index(e)] for e in sorted_emotions]
        for m in model_names
    }

    sorted_emotion_labels = [emotion_display[e] for e in sorted_emotions]

    print("Emotion sorted by KL divergence:", sorted_emotions)

    # ---------------------------------------------------
    # 5. Dot plot
    # ---------------------------------------------------
    plt.figure(figsize=(10, 6))

    for model in model_names:
        plt.scatter(
            sorted_x,
            sorted_kl[model],
            color=model_colors[model],
            s=120,
            label=legend_name_map[model]
        )

    plt.xticks(sorted_x, sorted_emotion_labels, fontsize=17, rotation=15)
    plt.ylabel("KL Divergence (emotion vs neutral)", fontsize=15)
    plt.ylim(0, 16)
    plt.tick_params(axis="y", labelsize=17)

    plt.title("Age", fontsize=18)

    plt.legend(
        fontsize=15,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.02, 1.0),
        framealpha=0.9
    )

    plt.tight_layout()
    plt.savefig("age_emotion.png", dpi=300, bbox_inches="tight")
    plt.show()


def age_influence2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from matplotlib.lines import Line2D

    sns.set_style("whitegrid")

    # ---------------------------------------------------
    # 1. Models, groups, colors, legend names
    # ---------------------------------------------------
    western_models = ["flux", "Proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "Wan2.1"]
    model_names = western_models + chinese_models

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

    # Western 圆形，Chinese 菱形
    model_markers = {
        "flux": "o",
        "Proteus": "o",
        "sd3": "o",
        "sana": "o",
        "hunyuan": "D",
        "qwen": "D",
        "kolors": "D",
        "Wan2.1": "D",
    }

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

    base_path = "/Users/wmt/projects/bias/data/jsons/age"

    # ---------------------------------------------------
    # 2. KL divergence
    # ---------------------------------------------------
    def kl_divergence(p, q, eps=1e-12):
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return np.sum(p * np.log(p / q))

    age_keys = ["0-9", "10-19", "20-39", "40-59", "60+"]
    emotions = ["surprised", "happy", "sad", "disgusted", "angry", "fearful"]

    emotion_display = {
        "surprised": "surprise",
        "happy": "happiness",
        "sad": "sadness",
        "disgusted": "disgust",
        "angry": "anger",
        "fearful": "fear",
    }

    # ---------------------------------------------------
    # 3. Compute KL(emotion || neutral)
    # ---------------------------------------------------
    kl_results = {m: [] for m in model_names}

    for model in model_names:
        with open(os.path.join(base_path, f"{model}_age_counts.json"), "r") as f:
            data = json.load(f)

        neutral_counts = data["person"]
        total_neutral = sum(neutral_counts.values())
        P_neutral = np.array([neutral_counts[k] / total_neutral for k in age_keys])

        for emo in emotions:
            emo_counts = data[emo]
            total_emo = sum(emo_counts.values())
            P_emo = np.array([emo_counts[k] / total_emo for k in age_keys])
            kl_results[model].append(kl_divergence(P_emo, P_neutral))

    # ---------------------------------------------------
    # 4. Sort emotions by mean KL
    # ---------------------------------------------------
    emotion_means = {
        emo: np.mean([kl_results[m][i] for m in model_names])
        for i, emo in enumerate(emotions)
    }

    sorted_emotions = sorted(emotions, key=lambda e: emotion_means[e])
    sorted_x = np.arange(len(sorted_emotions))

    sorted_kl = {
        m: [kl_results[m][emotions.index(e)] for e in sorted_emotions]
        for m in model_names
    }

    sorted_emotion_labels = [emotion_display[e] for e in sorted_emotions]

    # ---------------------------------------------------
    # 5. Layout: Western / Chinese x-offset within each emotion
    # ---------------------------------------------------
    n_emotions = len(sorted_emotions)
    dot_gap = 0.06  # gap between dots within a group
    group_gap = 0.18  # gap between Western and Chinese sub-groups
    n_w = len(western_models)
    n_c = len(chinese_models)

    western_offsets = np.arange(n_w) * dot_gap - (n_w - 1) * dot_gap / 2
    chinese_offsets = np.arange(n_c) * dot_gap + (n_w - 1) * dot_gap / 2 + group_gap - (n_c - 1) * dot_gap / 2

    def get_x(emo_idx, model):
        if model in western_models:
            return emo_idx + western_offsets[western_models.index(model)]
        else:
            return emo_idx + chinese_offsets[chinese_models.index(model)]

    # ---------------------------------------------------
    # 6. Plot
    # ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # 背景色区分 Western / Chinese
    w_center = np.mean(western_offsets)
    c_center = np.mean(chinese_offsets)
    half_w = (western_offsets[-1] - western_offsets[0]) / 2 + dot_gap * 0.8
    half_c = (chinese_offsets[-1] - chinese_offsets[0]) / 2 + dot_gap * 0.8

    for xi in sorted_x:
        ax.axvspan(xi + w_center - half_w, xi + w_center + half_w,
                   alpha=0.04, color="#4C8BE8", zorder=0)
        ax.axvspan(xi + c_center - half_c, xi + c_center + half_c,
                   alpha=0.04, color="#D94801", zorder=0)

    # Dots
    for model in model_names:
        for xi in sorted_x:
            ax.scatter(
                get_x(xi, model),
                sorted_kl[model][xi],
                color=model_colors[model],
                marker=model_markers[model],
                s=90,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
            )

    # ---------------------------------------------------
    # 7. Axes
    # ---------------------------------------------------
    ax.set_xticks(sorted_x + (w_center + c_center) / 2)
    ax.set_xticklabels(sorted_emotion_labels, fontsize=17, rotation=15)
    ax.set_ylabel("KL Divergence (emotion vs neutral)", fontsize=15)
    ax.set_ylim(0, 16)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_title("Age", fontsize=18)
    ax.set_xlim(sorted_x[0] - 0.4, sorted_x[-1] + 0.7)

    # ---------------------------------------------------
    # 8. Legend（分组，Western / Chinese 各一列）
    # ---------------------------------------------------
    def make_blank():
        return Line2D([0], [0], color="none")

    western_patch = mpatches.Patch(color="#4C8BE8", alpha=0.3)
    chinese_patch = mpatches.Patch(color="#D94801", alpha=0.3)

    w_handles = [
        plt.scatter([], [], color=model_colors[m], marker="o",
                    s=80, edgecolors="white", linewidths=0.4)
        for m in western_models
    ]
    c_handles = [
        plt.scatter([], [], color=model_colors[m], marker="D",
                    s=80, edgecolors="white", linewidths=0.4)
        for m in chinese_models
    ]

    final_handles = (
            [western_patch] + w_handles +
            [chinese_patch] + c_handles
    )
    final_labels = (
            ["Western"] + [legend_name_map[m] for m in western_models] +
            ["Chinese"] + [legend_name_map[m] for m in chinese_models]
    )

    leg = ax.legend(
        final_handles,
        final_labels,
        ncol=2,
        fontsize=13,
        loc="upper left",
        bbox_to_anchor=(0.02, 1.0),
        framealpha=0.9,
        handlelength=1.2,
        handleheight=1.0,
        borderpad=0.7,
        labelspacing=0.4,
        columnspacing=1.0,
    )

    for text in leg.get_texts():
        if text.get_text() in ("Western", "Chinese"):
            text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig("age_emotion.png", dpi=300, bbox_inches="tight")
    plt.show()

def gender_influence():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # ---------------------------------------------------
    # 1. Models, groups, colors, legend names
    # ---------------------------------------------------
    western_models = ["flux", "Proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "Wan2.1"]
    model_names = western_models + chinese_models

    model_colors = {
        "flux": "#08306B",
        "Proteus": "#2171B5",
        "sd3": "#C6DBEF",
        "sana": "#6BAED6",
        "hunyuan": "#7F2704",
        "qwen": "#E6550D",
        "kolors": "#A63603",
        "Wan2.1": "#FDBE85",
    }

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

    base_path = "/Users/wmt/projects/bias/data/jsons/gender"

    # ---------------------------------------------------
    # 2. KL divergence
    # ---------------------------------------------------
    def kl_divergence(p, q, eps=1e-12):
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return np.sum(p * np.log(p / q))

    gender_keys = ["Male", "Female"]

    emotions = ["surprised", "happy", "sad", "disgusted", "angry", "fearful"]

    # display name mapping（与 age 完全一致）
    emotion_display = {
        "surprised": "surprise",
        "happy": "happiness",
        "sad": "sadness",
        "disgusted": "disgust",
        "angry": "anger",
        "fearful": "fear",
    }

    # ---------------------------------------------------
    # 3. Compute KL(emotion || neutral)
    # ---------------------------------------------------
    kl_results = {m: [] for m in model_names}

    for model in model_names:
        with open(os.path.join(base_path, f"{model}_gender_counts.json"), "r") as f:
            data = json.load(f)

        neutral_counts = data["person"]
        total_neutral = sum(neutral_counts.values())
        P_neutral = np.array(
            [neutral_counts[k] / total_neutral for k in gender_keys]
        )

        for emo in emotions:
            emo_counts = data[emo]
            total_emo = sum(emo_counts.values())
            P_emo = np.array([emo_counts[k] / total_emo for k in gender_keys])

            KL = kl_divergence(P_emo, P_neutral)
            kl_results[model].append(KL)

    # ---------------------------------------------------
    # 4. Sort emotions by mean KL
    # ---------------------------------------------------
    emotion_means = {
        emo: np.mean([kl_results[m][i] for m in model_names])
        for i, emo in enumerate(emotions)
    }

    sorted_emotions = sorted(emotions, key=lambda e: emotion_means[e])
    sorted_x = np.arange(len(sorted_emotions))

    sorted_kl = {
        m: [kl_results[m][emotions.index(e)] for e in sorted_emotions]
        for m in model_names
    }

    sorted_emotion_labels = [emotion_display[e] for e in sorted_emotions]

    print("Emotion sorted by KL divergence (Gender):", sorted_emotions)

    # ---------------------------------------------------
    # 5. Dot plot
    # ---------------------------------------------------
    plt.figure(figsize=(10, 6))

    for model in model_names:
        plt.scatter(
            sorted_x,
            sorted_kl[model],
            color=model_colors[model],
            s=120,
            label=legend_name_map[model]
        )

    plt.xticks(sorted_x, sorted_emotion_labels, fontsize=17, rotation=15)
    plt.ylabel("KL Divergence (emotion vs neutral)", fontsize=15)

    # 👇 纵轴 scale 与 tick（你点名要的）
    plt.ylim(0, 6)
    plt.tick_params(axis="y", labelsize=17)

    plt.title("Gender", fontsize=18)

    plt.legend(
        fontsize=15,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.02, 1.0),
        framealpha=0.9
    )

    plt.tight_layout()
    plt.savefig("gender_emotion.png", dpi=300, bbox_inches="tight")
    plt.show()


def gender_influence2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from matplotlib.lines import Line2D

    sns.set_style("whitegrid")

    # ---------------------------------------------------
    # 1. Models, groups, colors, legend names
    # ---------------------------------------------------
    western_models = ["flux", "Proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "Wan2.1"]
    model_names = western_models + chinese_models

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

    # Western 圆形，Chinese 菱形
    model_markers = {
        "flux":    "o",
        "Proteus": "o",
        "sd3":     "o",
        "sana":    "o",
        "hunyuan": "D",
        "qwen":    "D",
        "kolors":  "D",
        "Wan2.1":  "D",
    }

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

    base_path = "/Users/wmt/projects/bias/data/jsons/gender"

    # ---------------------------------------------------
    # 2. KL divergence
    # ---------------------------------------------------
    def kl_divergence(p, q, eps=1e-12):
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return np.sum(p * np.log(p / q))

    gender_keys = ["Male", "Female"]
    emotions = ["surprised", "happy", "sad", "disgusted", "angry", "fearful"]

    emotion_display = {
        "surprised": "surprise",
        "happy":     "happiness",
        "sad":       "sadness",
        "disgusted": "disgust",
        "angry":     "anger",
        "fearful":   "fear",
    }

    # ---------------------------------------------------
    # 3. Compute KL(emotion || neutral)
    # ---------------------------------------------------
    kl_results = {m: [] for m in model_names}

    for model in model_names:
        with open(os.path.join(base_path, f"{model}_gender_counts.json"), "r") as f:
            data = json.load(f)

        neutral_counts = data["person"]
        total_neutral  = sum(neutral_counts.values())
        P_neutral = np.array([neutral_counts[k] / total_neutral for k in gender_keys])

        for emo in emotions:
            emo_counts = data[emo]
            total_emo  = sum(emo_counts.values())
            P_emo = np.array([emo_counts[k] / total_emo for k in gender_keys])
            kl_results[model].append(kl_divergence(P_emo, P_neutral))

    # ---------------------------------------------------
    # 4. Sort emotions by mean KL
    # ---------------------------------------------------
    emotion_means = {
        emo: np.mean([kl_results[m][i] for m in model_names])
        for i, emo in enumerate(emotions)
    }

    sorted_emotions = sorted(emotions, key=lambda e: emotion_means[e])
    sorted_x = np.arange(len(sorted_emotions))

    sorted_kl = {
        m: [kl_results[m][emotions.index(e)] for e in sorted_emotions]
        for m in model_names
    }

    sorted_emotion_labels = [emotion_display[e] for e in sorted_emotions]

    print("Emotion sorted by KL divergence (Gender):", sorted_emotions)

    # ---------------------------------------------------
    # 5. Layout: Western / Chinese x-offset within each emotion
    # ---------------------------------------------------
    dot_gap    = 0.06
    group_gap  = 0.18
    n_w        = len(western_models)
    n_c        = len(chinese_models)

    western_offsets = np.arange(n_w) * dot_gap - (n_w - 1) * dot_gap / 2
    chinese_offsets = np.arange(n_c) * dot_gap + (n_w - 1) * dot_gap / 2 + group_gap - (n_c - 1) * dot_gap / 2

    def get_x(emo_idx, model):
        if model in western_models:
            return emo_idx + western_offsets[western_models.index(model)]
        else:
            return emo_idx + chinese_offsets[chinese_models.index(model)]

    # ---------------------------------------------------
    # 6. Plot
    # ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # 背景色区分 Western / Chinese
    w_center = np.mean(western_offsets)
    c_center = np.mean(chinese_offsets)
    half_w   = (western_offsets[-1] - western_offsets[0]) / 2 + dot_gap * 0.8
    half_c   = (chinese_offsets[-1] - chinese_offsets[0]) / 2 + dot_gap * 0.8

    for xi in sorted_x:
        ax.axvspan(xi + w_center - half_w, xi + w_center + half_w,
                   alpha=0.04, color="#4C8BE8", zorder=0)
        ax.axvspan(xi + c_center - half_c, xi + c_center + half_c,
                   alpha=0.04, color="#D94801", zorder=0)

    # Dots
    for model in model_names:
        for xi in sorted_x:
            ax.scatter(
                get_x(xi, model),
                sorted_kl[model][xi],
                color=model_colors[model],
                marker=model_markers[model],
                s=90,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
            )

    # ---------------------------------------------------
    # 7. Axes
    # ---------------------------------------------------
    ax.set_xticks(sorted_x + (w_center + c_center) / 2)
    ax.set_xticklabels(sorted_emotion_labels, fontsize=17, rotation=15)
    ax.set_ylabel("KL Divergence (emotion vs neutral)", fontsize=15)
    ax.set_ylim(0, 6)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_title("Gender", fontsize=18)
    ax.set_xlim(sorted_x[0] - 0.4, sorted_x[-1] + 0.7)

    # ---------------------------------------------------
    # 8. Legend（分组，Western / Chinese 各一列）
    # ---------------------------------------------------
    def make_blank():
        return Line2D([0], [0], color="none")

    western_patch = mpatches.Patch(color="#4C8BE8", alpha=0.3)
    chinese_patch = mpatches.Patch(color="#D94801", alpha=0.3)

    w_handles = [
        plt.scatter([], [], color=model_colors[m], marker="o",
                    s=80, edgecolors="white", linewidths=0.4)
        for m in western_models
    ]
    c_handles = [
        plt.scatter([], [], color=model_colors[m], marker="D",
                    s=80, edgecolors="white", linewidths=0.4)
        for m in chinese_models
    ]

    final_handles = (
        [western_patch] + w_handles +
        [chinese_patch] + c_handles
    )
    final_labels = (
        ["Western"] + [legend_name_map[m] for m in western_models] +
        ["Chinese"] + [legend_name_map[m] for m in chinese_models]
    )

    leg = ax.legend(
        final_handles,
        final_labels,
        ncol=2,
        fontsize=13,
        loc="upper left",
        bbox_to_anchor=(0.02, 1.0),
        framealpha=0.9,
        handlelength=1.2,
        handleheight=1.0,
        borderpad=0.7,
        labelspacing=0.4,
        columnspacing=1.0,
    )

    for text in leg.get_texts():
        if text.get_text() in ("Western", "Chinese"):
            text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig("gender_emotion.png", dpi=300, bbox_inches="tight")
    plt.show()

def race_influence():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # ---------------------------------------------------
    # 1. Models, groups, colors, legend names
    # ---------------------------------------------------
    western_models = ["flux", "Proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "Wan2.1"]
    model_names = western_models + chinese_models

    model_colors = {
        "flux": "#08306B",
        "Proteus": "#2171B5",
        "sd3": "#C6DBEF",
        "sana": "#6BAED6",
        "hunyuan": "#7F2704",
        "qwen": "#E6550D",
        "kolors": "#A63603",
        "Wan2.1": "#FDBE85",
    }

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

    base_path = "/Users/wmt/projects/bias/data/jsons/race"

    # ---------------------------------------------------
    # 2. KL divergence
    # ---------------------------------------------------
    def kl_divergence(p, q, eps=1e-12):
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return np.sum(p * np.log(p / q))

    race_keys = ["White", "Black", "Asian", "Others"]
    emotions = ["surprised", "happy", "sad", "disgusted", "angry", "fearful"]

    # display name mapping（与 age / gender 完全一致）
    emotion_display = {
        "surprised": "surprise",
        "happy": "happiness",
        "sad": "sadness",
        "disgusted": "disgust",
        "angry": "anger",
        "fearful": "fear",
    }

    # ---------------------------------------------------
    # 3. Compute KL(emotion || neutral)
    # ---------------------------------------------------
    kl_results = {m: [] for m in model_names}

    for model in model_names:
        with open(os.path.join(base_path, f"{model}_emotion_race_counts.json"), "r") as f:
            data = json.load(f)

        neutral_counts = data["person"]
        total_neutral = sum(neutral_counts.values())
        P_neutral = np.array(
            [neutral_counts[k] / total_neutral for k in race_keys]
        )

        for emo in emotions:
            emo_counts = data[emo]
            total_emo = sum(emo_counts.values())
            P_emo = np.array([emo_counts[k] / total_emo for k in race_keys])

            KL = kl_divergence(P_emo, P_neutral)
            kl_results[model].append(KL)

    # ---------------------------------------------------
    # 4. Sort emotions by mean KL
    # ---------------------------------------------------
    emotion_means = {
        emo: np.mean([kl_results[m][i] for m in model_names])
        for i, emo in enumerate(emotions)
    }

    sorted_emotions = sorted(emotions, key=lambda e: emotion_means[e])
    sorted_x = np.arange(len(sorted_emotions))

    sorted_kl = {
        m: [kl_results[m][emotions.index(e)] for e in sorted_emotions]
        for m in model_names
    }

    sorted_emotion_labels = [emotion_display[e] for e in sorted_emotions]

    print("Emotion sorted by KL divergence (Race):", sorted_emotions)

    # ---------------------------------------------------
    # 5. Dot plot
    # ---------------------------------------------------
    plt.figure(figsize=(10, 6))

    for model in model_names:
        plt.scatter(
            sorted_x,
            sorted_kl[model],
            color=model_colors[model],
            s=120,
            label=legend_name_map[model]
        )

    plt.xticks(sorted_x, sorted_emotion_labels, fontsize=17, rotation=15)
    plt.ylabel("KL Divergence (emotion vs neutral)", fontsize=15)

    # 👇 纵轴 scale & tick（与你要求一致）
    plt.ylim(0, 6)
    plt.tick_params(axis="y", labelsize=17)

    plt.title("Race", fontsize=18)

    plt.legend(
        fontsize=15,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.02, 1.0),
        framealpha=0.9
    )

    plt.tight_layout()
    plt.savefig("race_emotion.png", dpi=300, bbox_inches="tight")
    plt.show()

def race_influence2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from matplotlib.lines import Line2D

    sns.set_style("whitegrid")

    # ---------------------------------------------------
    # 1. Models, groups, colors, legend names
    # ---------------------------------------------------
    western_models = ["flux", "Proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "Wan2.1"]
    model_names = western_models + chinese_models

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

    # Western 圆形，Chinese 菱形
    model_markers = {
        "flux":    "o",
        "Proteus": "o",
        "sd3":     "o",
        "sana":    "o",
        "hunyuan": "D",
        "qwen":    "D",
        "kolors":  "D",
        "Wan2.1":  "D",
    }

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

    base_path = "/Users/wmt/projects/bias/data/jsons/race"

    # ---------------------------------------------------
    # 2. KL divergence
    # ---------------------------------------------------
    def kl_divergence(p, q, eps=1e-12):
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return np.sum(p * np.log(p / q))

    race_keys = ["White", "Black", "Asian", "Others"]
    emotions  = ["surprised", "happy", "sad", "disgusted", "angry", "fearful"]

    emotion_display = {
        "surprised": "surprise",
        "happy":     "happiness",
        "sad":       "sadness",
        "disgusted": "disgust",
        "angry":     "anger",
        "fearful":   "fear",
    }

    # ---------------------------------------------------
    # 3. Compute KL(emotion || neutral)
    # ---------------------------------------------------
    kl_results = {m: [] for m in model_names}

    for model in model_names:
        with open(os.path.join(base_path, f"{model}_emotion_race_counts.json"), "r") as f:
            data = json.load(f)

        neutral_counts = data["person"]
        total_neutral  = sum(neutral_counts.values())
        P_neutral = np.array([neutral_counts[k] / total_neutral for k in race_keys])

        for emo in emotions:
            emo_counts = data[emo]
            total_emo  = sum(emo_counts.values())
            P_emo = np.array([emo_counts[k] / total_emo for k in race_keys])
            kl_results[model].append(kl_divergence(P_emo, P_neutral))

    # ---------------------------------------------------
    # 4. Sort emotions by mean KL
    # ---------------------------------------------------
    emotion_means = {
        emo: np.mean([kl_results[m][i] for m in model_names])
        for i, emo in enumerate(emotions)
    }

    sorted_emotions = sorted(emotions, key=lambda e: emotion_means[e])
    sorted_x = np.arange(len(sorted_emotions))

    sorted_kl = {
        m: [kl_results[m][emotions.index(e)] for e in sorted_emotions]
        for m in model_names
    }

    sorted_emotion_labels = [emotion_display[e] for e in sorted_emotions]

    print("Emotion sorted by KL divergence (Race):", sorted_emotions)

    # ---------------------------------------------------
    # 5. Layout: Western / Chinese x-offset within each emotion
    # ---------------------------------------------------
    dot_gap   = 0.06
    group_gap = 0.18
    n_w       = len(western_models)
    n_c       = len(chinese_models)

    western_offsets = np.arange(n_w) * dot_gap - (n_w - 1) * dot_gap / 2
    chinese_offsets = np.arange(n_c) * dot_gap + (n_w - 1) * dot_gap / 2 + group_gap - (n_c - 1) * dot_gap / 2

    def get_x(emo_idx, model):
        if model in western_models:
            return emo_idx + western_offsets[western_models.index(model)]
        else:
            return emo_idx + chinese_offsets[chinese_models.index(model)]

    # ---------------------------------------------------
    # 6. Plot
    # ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # 背景色区分 Western / Chinese
    w_center = np.mean(western_offsets)
    c_center = np.mean(chinese_offsets)
    half_w   = (western_offsets[-1] - western_offsets[0]) / 2 + dot_gap * 0.8
    half_c   = (chinese_offsets[-1] - chinese_offsets[0]) / 2 + dot_gap * 0.8

    for xi in sorted_x:
        ax.axvspan(xi + w_center - half_w, xi + w_center + half_w,
                   alpha=0.04, color="#4C8BE8", zorder=0)
        ax.axvspan(xi + c_center - half_c, xi + c_center + half_c,
                   alpha=0.04, color="#D94801", zorder=0)

    # Dots
    for model in model_names:
        for xi in sorted_x:
            ax.scatter(
                get_x(xi, model),
                sorted_kl[model][xi],
                color=model_colors[model],
                marker=model_markers[model],
                s=90,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
            )

    # ---------------------------------------------------
    # 7. Axes
    # ---------------------------------------------------
    ax.set_xticks(sorted_x + (w_center + c_center) / 2)
    ax.set_xticklabels(sorted_emotion_labels, fontsize=17, rotation=15)
    ax.set_ylabel("KL Divergence (emotion vs neutral)", fontsize=15)
    ax.set_ylim(0, 6)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_title("Race", fontsize=18)
    ax.set_xlim(sorted_x[0] - 0.4, sorted_x[-1] + 0.7)

    # ---------------------------------------------------
    # 8. Legend（分组，Western / Chinese 各一列）
    # ---------------------------------------------------
    western_patch = mpatches.Patch(color="#4C8BE8", alpha=0.3)
    chinese_patch = mpatches.Patch(color="#D94801", alpha=0.3)

    w_handles = [
        plt.scatter([], [], color=model_colors[m], marker="o",
                    s=80, edgecolors="white", linewidths=0.4)
        for m in western_models
    ]
    c_handles = [
        plt.scatter([], [], color=model_colors[m], marker="D",
                    s=80, edgecolors="white", linewidths=0.4)
        for m in chinese_models
    ]

    final_handles = (
        [western_patch] + w_handles +
        [chinese_patch] + c_handles
    )
    final_labels = (
        ["Western"] + [legend_name_map[m] for m in western_models] +
        ["Chinese"] + [legend_name_map[m] for m in chinese_models]
    )

    leg = ax.legend(
        final_handles,
        final_labels,
        ncol=2,
        fontsize=13,
        loc="upper left",
        bbox_to_anchor=(0.02, 1.0),
        framealpha=0.9,
        handlelength=1.2,
        handleheight=1.0,
        borderpad=0.7,
        labelspacing=0.4,
        columnspacing=1.0,
    )

    for text in leg.get_texts():
        if text.get_text() in ("Western", "Chinese"):
            text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig("race_emotion.png", dpi=300, bbox_inches="tight")
    plt.show()

def attractive_influence():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    # ---------------------------------------------------
    # 1. Models, groups, colors, legend names
    # ---------------------------------------------------
    western_models = ["flux", "Proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "Wan2.1"]
    model_names = western_models + chinese_models

    model_colors = {
        "flux": "#08306B",
        "Proteus": "#2171B5",
        "sd3": "#C6DBEF",
        "sana": "#6BAED6",
        "hunyuan": "#7F2704",
        "qwen": "#E6550D",
        "kolors": "#A63603",
        "Wan2.1": "#FDBE85",
    }

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

    base_path = "/Users/wmt/projects/bias/data/jsons/attract"

    # ---------------------------------------------------
    # 2. KL divergence
    # ---------------------------------------------------
    def kl_divergence(p, q, eps=1e-12):
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return np.sum(p * np.log(p / q))

    attr_keys = ["low", "medium", "high"]
    emotions = ["surprised", "happy", "sad", "disgusted", "angry", "fearful"]

    # display name mapping（与其他 influence 图完全一致）
    emotion_display = {
        "surprised": "surprise",
        "happy": "happiness",
        "sad": "sadness",
        "disgusted": "disgust",
        "angry": "anger",
        "fearful": "fear",
    }

    # ---------------------------------------------------
    # 3. Compute KL(emotion || neutral)
    # ---------------------------------------------------
    kl_results = {m: [] for m in model_names}

    for model in model_names:
        with open(os.path.join(base_path, f"{model}_attractiveness_counts.json"), "r") as f:
            data = json.load(f)

        neutral_counts = data["person"]
        total_neutral = sum(neutral_counts.values())
        P_neutral = np.array(
            [neutral_counts[a] / total_neutral for a in attr_keys]
        )

        for emo in emotions:
            emo_counts = data[emo]
            total_emo = sum(emo_counts.values())
            P_emo = np.array([emo_counts[a] / total_emo for a in attr_keys])

            KL = kl_divergence(P_emo, P_neutral)
            kl_results[model].append(KL)

    # ---------------------------------------------------
    # 4. Sort emotions by mean KL
    # ---------------------------------------------------
    emotion_means = {
        emo: np.mean([kl_results[m][i] for m in model_names])
        for i, emo in enumerate(emotions)
    }

    sorted_emotions = sorted(emotions, key=lambda e: emotion_means[e])
    sorted_x = np.arange(len(sorted_emotions))

    sorted_kl = {
        m: [kl_results[m][emotions.index(e)] for e in sorted_emotions]
        for m in model_names
    }

    sorted_emotion_labels = [emotion_display[e] for e in sorted_emotions]

    print("Emotion sorted by KL divergence (Attractiveness):", sorted_emotions)

    # ---------------------------------------------------
    # 5. Dot plot
    # ---------------------------------------------------
    plt.figure(figsize=(10, 6))

    for model in model_names:
        plt.scatter(
            sorted_x,
            sorted_kl[model],
            color=model_colors[model],
            s=120,
            label=legend_name_map[model]
        )

    plt.xticks(sorted_x, sorted_emotion_labels, fontsize=17, rotation=15)
    plt.ylabel("KL Divergence (emotion vs neutral)", fontsize=15)

    # 👇 y-scale 保持不变
    plt.ylim(0, 16)
    plt.tick_params(axis="y", labelsize=17)

    plt.title("Attractiveness", fontsize=18)

    plt.legend(
        fontsize=15,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.02, 1.0),
        framealpha=0.9
    )

    plt.tight_layout()
    plt.savefig("attractiveness_emotion.png", dpi=300, bbox_inches="tight")
    plt.show()


def attractive_influence2():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from matplotlib.lines import Line2D

    sns.set_style("whitegrid")

    # ---------------------------------------------------
    # 1. Models, groups, colors, legend names
    # ---------------------------------------------------
    western_models = ["flux", "Proteus", "sd3", "sana"]
    chinese_models = ["hunyuan", "qwen", "kolors", "Wan2.1"]
    model_names = western_models + chinese_models

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

    # Western 圆形，Chinese 菱形
    model_markers = {
        "flux":    "o",
        "Proteus": "o",
        "sd3":     "o",
        "sana":    "o",
        "hunyuan": "D",
        "qwen":    "D",
        "kolors":  "D",
        "Wan2.1":  "D",
    }

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

    base_path = "/Users/wmt/projects/bias/data/jsons/attract"

    # ---------------------------------------------------
    # 2. KL divergence
    # ---------------------------------------------------
    def kl_divergence(p, q, eps=1e-12):
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return np.sum(p * np.log(p / q))

    attr_keys = ["low", "medium", "high"]
    emotions  = ["surprised", "happy", "sad", "disgusted", "angry", "fearful"]

    emotion_display = {
        "surprised": "surprise",
        "happy":     "happiness",
        "sad":       "sadness",
        "disgusted": "disgust",
        "angry":     "anger",
        "fearful":   "fear",
    }

    # ---------------------------------------------------
    # 3. Compute KL(emotion || neutral)
    # ---------------------------------------------------
    kl_results = {m: [] for m in model_names}

    for model in model_names:
        with open(os.path.join(base_path, f"{model}_attractiveness_counts.json"), "r") as f:
            data = json.load(f)

        neutral_counts = data["person"]
        total_neutral  = sum(neutral_counts.values())
        P_neutral = np.array([neutral_counts[a] / total_neutral for a in attr_keys])

        for emo in emotions:
            emo_counts = data[emo]
            total_emo  = sum(emo_counts.values())
            P_emo = np.array([emo_counts[a] / total_emo for a in attr_keys])
            kl_results[model].append(kl_divergence(P_emo, P_neutral))

    # ---------------------------------------------------
    # 4. Sort emotions by mean KL
    # ---------------------------------------------------
    emotion_means = {
        emo: np.mean([kl_results[m][i] for m in model_names])
        for i, emo in enumerate(emotions)
    }

    sorted_emotions = sorted(emotions, key=lambda e: emotion_means[e])
    sorted_x = np.arange(len(sorted_emotions))

    sorted_kl = {
        m: [kl_results[m][emotions.index(e)] for e in sorted_emotions]
        for m in model_names
    }

    sorted_emotion_labels = [emotion_display[e] for e in sorted_emotions]

    print("Emotion sorted by KL divergence (Attractiveness):", sorted_emotions)

    # ---------------------------------------------------
    # 5. Layout: Western / Chinese x-offset within each emotion
    # ---------------------------------------------------
    dot_gap   = 0.06
    group_gap = 0.18
    n_w       = len(western_models)
    n_c       = len(chinese_models)

    western_offsets = np.arange(n_w) * dot_gap - (n_w - 1) * dot_gap / 2
    chinese_offsets = np.arange(n_c) * dot_gap + (n_w - 1) * dot_gap / 2 + group_gap - (n_c - 1) * dot_gap / 2

    def get_x(emo_idx, model):
        if model in western_models:
            return emo_idx + western_offsets[western_models.index(model)]
        else:
            return emo_idx + chinese_offsets[chinese_models.index(model)]

    # ---------------------------------------------------
    # 6. Plot
    # ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # 背景色区分 Western / Chinese
    w_center = np.mean(western_offsets)
    c_center = np.mean(chinese_offsets)
    half_w   = (western_offsets[-1] - western_offsets[0]) / 2 + dot_gap * 0.8
    half_c   = (chinese_offsets[-1] - chinese_offsets[0]) / 2 + dot_gap * 0.8

    for xi in sorted_x:
        ax.axvspan(xi + w_center - half_w, xi + w_center + half_w,
                   alpha=0.04, color="#4C8BE8", zorder=0)
        ax.axvspan(xi + c_center - half_c, xi + c_center + half_c,
                   alpha=0.04, color="#D94801", zorder=0)

    # Dots
    for model in model_names:
        for xi in sorted_x:
            ax.scatter(
                get_x(xi, model),
                sorted_kl[model][xi],
                color=model_colors[model],
                marker=model_markers[model],
                s=90,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
            )

    # ---------------------------------------------------
    # 7. Axes
    # ---------------------------------------------------
    ax.set_xticks(sorted_x + (w_center + c_center) / 2)
    ax.set_xticklabels(sorted_emotion_labels, fontsize=17, rotation=15)
    ax.set_ylabel("KL Divergence (emotion vs neutral)", fontsize=15)
    ax.set_ylim(0, 16)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_title("Attractiveness", fontsize=18)
    ax.set_xlim(sorted_x[0] - 0.4, sorted_x[-1] + 0.7)

    # ---------------------------------------------------
    # 8. Legend（分组，Western / Chinese 各一列）
    # ---------------------------------------------------
    western_patch = mpatches.Patch(color="#4C8BE8", alpha=0.3)
    chinese_patch = mpatches.Patch(color="#D94801", alpha=0.3)

    w_handles = [
        plt.scatter([], [], color=model_colors[m], marker="o",
                    s=80, edgecolors="white", linewidths=0.4)
        for m in western_models
    ]
    c_handles = [
        plt.scatter([], [], color=model_colors[m], marker="D",
                    s=80, edgecolors="white", linewidths=0.4)
        for m in chinese_models
    ]

    final_handles = (
        [western_patch] + w_handles +
        [chinese_patch] + c_handles
    )
    final_labels = (
        ["Western"] + [legend_name_map[m] for m in western_models] +
        ["Chinese"] + [legend_name_map[m] for m in chinese_models]
    )

    leg = ax.legend(
        final_handles,
        final_labels,
        ncol=2,
        fontsize=13,
        loc="upper left",
        bbox_to_anchor=(0.02, 1.0),
        framealpha=0.9,
        handlelength=1.2,
        handleheight=1.0,
        borderpad=0.7,
        labelspacing=0.4,
        columnspacing=1.0,
    )

    for text in leg.get_texts():
        if text.get_text() in ("Western", "Chinese"):
            text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig("attractiveness_emotion.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    age_influence2()
    gender_influence2()
    race_influence2()
    attractive_influence2()