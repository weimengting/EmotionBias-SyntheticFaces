import os
import json
import numpy as np
from scipy.stats import chisquare
from statsmodels.stats.multitest import multipletests


def compute_results():
    # ===== 路径 =====
    base_dirs = {
        "age": "/Users/wmt/projects/bias/write_into_json/age",
        "gender": "/Users/wmt/projects/bias/write_into_json/gender",
        "race": "/Users/wmt/projects/bias/write_into_json/race",
        "attractiveness": "/Users/wmt/projects/bias/write_into_json/attractiveness"
    }

    # ===== 类别定义 =====
    category_keys = {
        "age": ["0-9", "10-19", "20-39", "40-59", "60+"],
        "gender": ["Male", "Female"],
        "race": ["White", "Black", "Asian", "Others"],
        "attractiveness": ["0", "1", "2"]
    }

    # ===== 模型列表 =====
    models = [
        "flux", "proteus", "sd3", "sana",
        "hunyuan", "qwen", "kolors", "wan2.1"
    ]

    ignore_emotion = "unhappy"
    alpha = 0.001

    print(f"Nominal alpha: {alpha}")

    # ===== 收集所有检验结果 =====
    all_pvals = []
    all_meta = []

    for attr, base_dir in base_dirs.items():
        keys = category_keys[attr]

        for model in models:
            file_path = os.path.join(base_dir, f"{model}_{attr}.json")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # neutral
            neutral_dict = data["neutral"]
            neutral_counts = np.array(
                [neutral_dict.get(k, 0) for k in keys],
                dtype=float
            )

            neutral_sum = neutral_counts.sum()
            if neutral_sum == 0:
                print(f"[Warning] neutral sum is 0: {file_path}")
                continue

            neutral_prob = neutral_counts / neutral_sum

            for emotion, counts_dict in data.items():
                if emotion in ["neutral", ignore_emotion]:
                    continue

                observed = np.array(
                    [counts_dict.get(k, 0) for k in keys],
                    dtype=float
                )

                obs_sum = observed.sum()
                if obs_sum == 0:
                    print(f"[Warning] observed sum is 0: {model} / {attr} / {emotion}")
                    continue

                expected = neutral_prob * obs_sum

                eps = 1e-8
                expected = np.clip(expected, eps, None)
                expected = expected * (obs_sum / expected.sum())

                chi2, p = chisquare(f_obs=observed, f_exp=expected)

                all_pvals.append(p)
                all_meta.append({
                    "model": model,
                    "attr": attr,
                    "emotion": emotion,
                    "chi2": chi2,
                    "p": p
                })

    # ===== Bonferroni 校正 =====
    reject, pvals_corrected, _, _ = multipletests(all_pvals, alpha=alpha, method="bonferroni")

    # ===== 统计 overall =====
    total_count = len(reject)
    significant_count = int(np.sum(reject))
    overall_percent = 100 * significant_count / total_count

    print(f"\nBonferroni-corrected overall: {overall_percent:.2f}% significant")

    # ===== 分 attribute 统计 =====
    attr_total = {k: 0 for k in base_dirs}
    attr_sig = {k: 0 for k in base_dirs}

    for meta, is_sig, p_corr in zip(all_meta, reject, pvals_corrected):
        attr = meta["attr"]
        attr_total[attr] += 1
        if is_sig:
            attr_sig[attr] += 1
        meta["p_corrected"] = p_corr
        meta["significant"] = bool(is_sig)

    print("\nPer attribute:")
    for attr in base_dirs:
        if attr_total[attr] == 0:
            print(f"{attr}: no valid tests")
        else:
            percent = 100 * attr_sig[attr] / attr_total[attr]
            print(f"{attr}: {percent:.2f}%")

    # ===== 可选：打印不显著的组合 =====
    print("\nNon-significant tests after Bonferroni:")
    for meta in all_meta:
        if not meta["significant"]:
            print(
                f"{meta['model']} / {meta['attr']} / {meta['emotion']} | "
                f"raw p={meta['p']:.3e}, corrected p={meta['p_corrected']:.3e}"
            )
        # ===== 新增：按 attr × emotion 统计显著比例 =====
        # 统计每个 (attr, emotion) 组合中有多少模型显著
        from collections import defaultdict
        combo_total = defaultdict(int)
        combo_sig = defaultdict(int)

        for meta in all_meta:
            key = (meta["attr"], meta["emotion"])
            combo_total[key] += 1
            if meta["significant"]:
                combo_sig[key] += 1

        # 整理成 attr × emotion 矩阵
        all_emotions = sorted(set(m["emotion"] for m in all_meta))
        all_attrs = list(base_dirs.keys())

        print("\nPer attr × emotion (% significant):")
        matrix = {}
        for attr in all_attrs:
            matrix[attr] = {}
            for emotion in all_emotions:
                key = (attr, emotion)
                if combo_total[key] == 0:
                    matrix[attr][emotion] = np.nan
                else:
                    matrix[attr][emotion] = 100 * combo_sig[key] / combo_total[key]
            print(f"{attr}: { {e: f'{matrix[attr][e]:.1f}%' for e in all_emotions} }")

        return matrix, all_attrs, all_emotions, overall_percent



def plot_results():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # ===== 数据 =====
    df = pd.DataFrame({
        "Attribute": ["Age", "Gender", "Race", "Attractiveness"],
        "Significant (%)": [95.83, 83.33, 97.92, 97.92]
    })

    # ===== 画图 =====
    sns.set(style="whitegrid", font_scale=1.1)

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.barplot(
        data=df,
        x="Attribute",
        y="Significant (%)",
        palette="muted",
        ax=ax
    )

    # ===== 参考线：overall =====
    ax.axhline(y=93.75, color="gray", linestyle="--", linewidth=1.2,
               label="Overall (93.75%)", zorder=0)  # ← 加 zorder=0

    # ===== 数值标签 =====
    for patch, pct in zip(ax.patches, df["Significant (%)"]):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.8,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=10
        )

    # ===== 坐标轴 =====
    ax.set_ylim(0, 110)
    ax.set_ylabel("Significant Tests (%)")
    ax.set_xlabel("")
    ax.set_title("Proportion of Significant Tests after Bonferroni Correction\n"
                 r"($\chi^2$ test, $\alpha^*$ = 0.001/192)", fontsize=10)

    ax.legend(fontsize=9, bbox_to_anchor=(1, 1.06), loc="upper right")
    sns.despine()

    plt.tight_layout()
    plt.savefig("significant_proportion.png", dpi=300)
    plt.show()


def plot_results2(matrix, all_attrs, all_emotions, overall_percent):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # ===== 整理数据 =====
    attr_labels = ["Age", "Gender", "Race", "Attractiveness"]
    emotion_labels = [e.capitalize() for e in all_emotions]

    data_matrix = np.array([
        [matrix[attr][emotion] for emotion in all_emotions]
        for attr in all_attrs
    ])

    df_heat = pd.DataFrame(data_matrix, index=attr_labels, columns=emotion_labels)

    # ===== 方案1：热力图 =====
    fig1, ax1 = plt.subplots(figsize=(9, 4))

    sns.heatmap(
        df_heat,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        vmin=0, vmax=100,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Significant Tests (%)"},
        ax=ax1
    )

    ax1.set_title(
        "Proportion of Significant Tests after Bonferroni Correction\n"
        r"($\chi^2$ test, $\alpha^*$ = 0.001/192)",
        fontsize=11
    )
    ax1.set_xlabel("Emotion")
    ax1.set_ylabel("Attribute")
    ax1.tick_params(axis="x", rotation=0)
    ax1.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig("heatmap_emotion_attr.png", dpi=300)
    plt.show()

    # ===== 方案2：分组柱状图 =====
    df_bar = df_heat.reset_index().melt(
        id_vars="index", var_name="Emotion", value_name="Significant (%)"
    ).rename(columns={"index": "Attribute"})

    sns.set(style="whitegrid", font_scale=1.0)
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=df_bar,
        x="Attribute",
        y="Significant (%)",
        hue="Emotion",
        palette="muted",
        ax=ax2
    )

    ax2.axhline(y=overall_percent, color="gray", linestyle="--", linewidth=1.2,
                label=f"Overall ({overall_percent:.2f}%)", zorder=0)

    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Significant Tests (%)")
    ax2.set_xlabel("")
    ax2.set_title(
        "Proportion of Significant Tests after Bonferroni Correction\n"
        r"($\chi^2$ test, $\alpha^*$ = 0.001/192)",
        fontsize=11
    )
    ax2.legend(title="Emotion", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    sns.despine()

    plt.tight_layout()
    plt.savefig("barplot_emotion_attr.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    matrix, all_attrs, all_emotions, overall_percent = compute_results()
    plot_results2(matrix, all_attrs, all_emotions, overall_percent)
    print("done")