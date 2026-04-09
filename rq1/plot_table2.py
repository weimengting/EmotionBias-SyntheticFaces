import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def attribute_separate():
    models = ["Flux", "Proteus", "SD3", "SANA", "Hunyuan", "Qwen", "Kolors", "Wan2.1"]

    data_dict = {
        "Gender": {
            "KL":  [0.04, 1.07, 0.28, 0.85, 0.27, 0.40, 0.00, 0.00],
            "JS":  [0.01, 0.16, 0.06, 0.14, 0.06, 0.08, 0.00, 0.00],
            "TVD": [0.14, 0.47, 0.33, 0.45, 0.32, 0.37, 0.04, 0.03],
        },
        "Race": {
            "KL":  [0.16, 9.45, 0.11, 5.07, 1.79, 0.21, 1.96, 0.36],
            "JS":  [0.04, 0.40, 0.03, 0.30, 0.26, 0.04, 0.33, 0.09],
            "TVD": [0.24, 0.81, 0.23, 0.72, 0.64, 0.23, 0.74, 0.39],
        },
        "Age": {
            "KL":  [0.86, 8.61, 8.52, 13.69, 8.02, 5.58, 1.09, 5.59],
            "JS":  [0.12, 0.26, 0.29, 0.26,  0.20, 0.22, 0.19, 0.28],
            "TVD": [0.40, 0.62, 0.67, 0.54,  0.53, 0.52, 0.56, 0.67],
        },
    }

    ci_dict = {
        "Gender": {
            "KL":  [(0.02,0.06),(0.92,1.26),(0.23,0.33),(0.73,1.01),(0.22,0.33),(0.34,0.48),(0.00,0.01),(0.00,0.01)],
            "JS":  [(0.01,0.01),(0.15,0.18),(0.05,0.07),(0.13,0.16),(0.05,0.07),(0.07,0.10),(0.00,0.00),(0.00,0.00)],
            "TVD": [(0.11,0.16),(0.46,0.48),(0.30,0.35),(0.44,0.47),(0.30,0.35),(0.35,0.39),(0.00,0.06),(0.00,0.05)],
        },
        "Race": {
            "KL":  [(0.13,0.20),(9.32,9.66),(0.09,0.15),(4.98,5.18),(1.55,5.31),(0.17,0.27),(1.74,5.54),(0.31,0.42)],
            "JS":  [(0.03,0.05),(0.38,0.41),(0.02,0.04),(0.28,0.32),(0.24,0.27),(0.04,0.05),(0.31,0.35),(0.08,0.10)],
            "TVD": [(0.21,0.27),(0.79,0.82),(0.20,0.26),(0.70,0.74),(0.62,0.65),(0.21,0.25),(0.72,0.76),(0.36,0.41)],
        },
        "Age": {
            "KL":  [(0.70,4.44),(8.41,12.09),(8.41,8.67),(13.67,13.70),(7.96,8.09),(5.30,11.92),(0.95,1.39),(5.40,8.63)],
            "JS":  [(0.11,0.13),(0.25,0.27),(0.27,0.30),(0.25,0.26),(0.19,0.22),(0.21,0.23),(0.18,0.21),(0.27,0.30)],
            "TVD": [(0.37,0.42),(0.60,0.63),(0.65,0.68),(0.54,0.54),(0.50,0.55),(0.50,0.55),(0.54,0.58),(0.65,0.68)],
        },
    }

    # =========================
    # 2. 转成 long format DataFrame
    # =========================
    rows = []
    for attr in ["Gender", "Race", "Age"]:
        for metric in ["KL", "JS", "TVD"]:
            for j, model in enumerate(models):
                val = data_dict[attr][metric][j]
                lo, hi = ci_dict[attr][metric][j]
                rows.append({
                    "Attribute": attr,
                    "Metric":    metric,
                    "Model":     model,
                    "Value":     val,
                    "CI_low":    val - lo,
                    "CI_high":   hi - val,
                })

    df = pd.DataFrame(rows)

    # =========================
    # 3. Plot
    # =========================
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = {"KL": "#4C8BE8", "JS": "#F5A623", "TVD": "#2CA02C"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, attr in zip(axes, ["Gender", "Race", "Age"]):
        sub = df[df["Attribute"] == attr]

        # 背景色区分 Western / Chinese
        ax.axvspan(-0.5, 3.5, alpha=0.06, color="#4C8BE8", zorder=0)
        ax.axvspan( 3.5, 7.5, alpha=0.06, color="#F5734C", zorder=0)

        # seaborn barplot
        sns.barplot(
            data=sub,
            x="Model", y="Value", hue="Metric",
            palette=palette,
            ax=ax,
            errorbar=None,       # 我们手动加CI
            width=0.7,
        )

        # 手动添加 CI 误差棒
        n_metrics = 3
        n_models  = len(models)
        # seaborn 的 bar 宽度和偏移需要手动算
        total_width = 0.7
        bar_width   = total_width / n_metrics
        offsets     = [-bar_width, 0, bar_width]

        for i, metric in enumerate(["KL", "JS", "TVD"]):
            sub_m = sub[sub["Metric"] == metric].reset_index(drop=True)
            xs = np.arange(n_models) + offsets[i]
            ax.errorbar(
                xs, sub_m["Value"],
                yerr=[sub_m["CI_low"], sub_m["CI_high"]],
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=3,
            )

        # 标注 worst ▲ / best ★
        for i, metric in enumerate(["KL", "JS", "TVD"]):
            sub_m = sub[sub["Metric"] == metric].reset_index(drop=True)
            vals  = sub_m["Value"].values
            xs    = np.arange(n_models) + offsets[i]
            worst_idx = int(np.argmax(vals))
            best_idx  = int(np.argmin(vals))
            ax.annotate("▲", xy=(xs[worst_idx], vals[worst_idx]),
                        ha="center", va="bottom", fontsize=8, color="red")
            ax.annotate("★", xy=(xs[best_idx],  vals[best_idx]),
                        ha="center", va="bottom", fontsize=8, color="green")

        ax.set_title(attr, fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score (lower is better)", fontsize=10)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
        ax.get_legend().remove()

    # 统一图例
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in palette.values()]
    fig.legend(handles, palette.keys(),
               title="Metric", loc="lower center",
               ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.05))

    fig.text(0.5, -0.10,
             "▲ worst per metric   ★ best per metric   "
             "Blue background = Western models   Red background = Chinese models",
             ha="center", fontsize=9, color="gray")

    # plt.suptitle("Bias Metrics for Gender, Race, and Age\nRelative to Global Population Distributions",
    #              fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("bias_metrics_seaborn.pdf", dpi=300, bbox_inches="tight")
    plt.show()


def metric_separate():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns

    # =========================
    # 1. 数据
    # =========================
    models = ["Flux", "Proteus", "SD3", "SANA", "Hunyuan", "Qwen", "Kolors", "Wan2.1"]

    data_dict = {
        "Gender": {
            "KL": [0.04, 1.07, 0.28, 0.85, 0.27, 0.40, 0.00, 0.00],
            "JS": [0.01, 0.16, 0.06, 0.14, 0.06, 0.08, 0.00, 0.00],
            "TVD": [0.14, 0.47, 0.33, 0.45, 0.32, 0.37, 0.04, 0.03],
        },
        "Race": {
            "KL": [0.16, 9.45, 0.11, 5.07, 1.79, 0.21, 1.96, 0.36],
            "JS": [0.04, 0.40, 0.03, 0.30, 0.26, 0.04, 0.33, 0.09],
            "TVD": [0.24, 0.81, 0.23, 0.72, 0.64, 0.23, 0.74, 0.39],
        },
        "Age": {
            "KL": [0.86, 8.61, 8.52, 13.69, 8.02, 5.58, 1.09, 5.59],
            "JS": [0.12, 0.26, 0.29, 0.26, 0.20, 0.22, 0.19, 0.28],
            "TVD": [0.40, 0.62, 0.67, 0.54, 0.53, 0.52, 0.56, 0.67],
        },
    }

    ci_dict = {
        "Gender": {
            "KL": [(0.02, 0.06), (0.92, 1.26), (0.23, 0.33), (0.73, 1.01), (0.22, 0.33), (0.34, 0.48), (0.00, 0.01),
                   (0.00, 0.01)],
            "JS": [(0.01, 0.01), (0.15, 0.18), (0.05, 0.07), (0.13, 0.16), (0.05, 0.07), (0.07, 0.10), (0.00, 0.00),
                   (0.00, 0.00)],
            "TVD": [(0.11, 0.16), (0.46, 0.48), (0.30, 0.35), (0.44, 0.47), (0.30, 0.35), (0.35, 0.39), (0.00, 0.06),
                    (0.00, 0.05)],
        },
        "Race": {
            "KL": [(0.13, 0.20), (9.32, 9.66), (0.09, 0.15), (4.98, 5.18), (1.55, 5.31), (0.17, 0.27), (1.74, 5.54),
                   (0.31, 0.42)],
            "JS": [(0.03, 0.05), (0.38, 0.41), (0.02, 0.04), (0.28, 0.32), (0.24, 0.27), (0.04, 0.05), (0.31, 0.35),
                   (0.08, 0.10)],
            "TVD": [(0.21, 0.27), (0.79, 0.82), (0.20, 0.26), (0.70, 0.74), (0.62, 0.65), (0.21, 0.25), (0.72, 0.76),
                    (0.36, 0.41)],
        },
        "Age": {
            "KL": [(0.70, 4.44), (8.41, 12.09), (8.41, 8.67), (13.67, 13.70), (7.96, 8.09), (5.30, 11.92), (0.95, 1.39),
                   (5.40, 8.63)],
            "JS": [(0.11, 0.13), (0.25, 0.27), (0.27, 0.30), (0.25, 0.26), (0.19, 0.22), (0.21, 0.23), (0.18, 0.21),
                   (0.27, 0.30)],
            "TVD": [(0.37, 0.42), (0.60, 0.63), (0.65, 0.68), (0.54, 0.54), (0.50, 0.55), (0.50, 0.55), (0.54, 0.58),
                    (0.65, 0.68)],
        },
    }

    metrics = ["KL", "JS", "TVD"]
    attrs = ["Gender", "Race", "Age"]

    # 每个 attribute 一个颜色
    ATTR_COLORS = {"Gender": "#378add", "Race": "#ef9f27", "Age": "#639922"}

    # =========================
    # 2. 布局参数
    # 每张图: x轴是 model, 每个 model 有3根 bar (Gender/Race/Age) 并排
    # =========================
    n_models = len(models)
    n_attrs = len(attrs)
    bar_width = 0.22
    group_gap = 0.08  # 每组内部间距
    offsets = np.array([-1, 0, 1]) * (bar_width + group_gap / 2)

    sns.set_theme(style="whitegrid", font_scale=1.05)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(wspace=0.35)

    xs = np.arange(n_models)

    for ax, metric in zip(axes, metrics):
        # 背景色区分 Western / Chinese
        ax.axvspan(-0.5, 3.5, alpha=0.07, color="#378add", zorder=0)
        ax.axvspan(3.5, 7.5, alpha=0.07, color="#d85a30", zorder=0)

        for i, attr in enumerate(attrs):
            vals = np.array(data_dict[attr][metric])
            cis = ci_dict[attr][metric]
            err_low = np.array([v - lo for v, (lo, _) in zip(vals, cis)])
            err_high = np.array([hi - v for v, (_, hi) in zip(vals, cis)])

            xpos = xs + offsets[i]

            ax.bar(xpos, vals, width=bar_width,
                   color=ATTR_COLORS[attr], label=attr,
                   zorder=2, linewidth=0, alpha=0.85)

            ax.errorbar(xpos, vals,
                        yerr=[err_low, err_high],
                        fmt="none", ecolor="black",
                        elinewidth=0.8, capsize=2.5, zorder=3)

            # ▲ worst  ★ best
            worst_idx = int(np.argmax(vals))
            best_idx = int(np.argmin(vals))
            y_pad = (vals + err_high).max() * 0.03

            ax.annotate("▲",
                        xy=(xpos[worst_idx], vals[worst_idx] + err_high[worst_idx] + y_pad),
                        ha="center", va="bottom", fontsize=7,
                        color=ATTR_COLORS[attr])
            ax.annotate("★",
                        xy=(xpos[best_idx], vals[best_idx] + err_high[best_idx] + y_pad),
                        ha="center", va="bottom", fontsize=7,
                        color=ATTR_COLORS[attr])

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel("Score (lower is better)", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlim(-0.6, n_models - 0.4)

    # =========================
    # 3. 图例
    # =========================
    attr_handles = [mpatches.Patch(color=ATTR_COLORS[a], label=a) for a in attrs]
    bg_handles = [
        mpatches.Patch(color="#378add", alpha=0.25, label="Western models (bg)"),
        mpatches.Patch(color="#d85a30", alpha=0.25, label="Chinese models (bg)"),
    ]
    fig.legend(handles=attr_handles + bg_handles,
               loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.04),
               frameon=True)

    fig.text(0.5, -0.09,
             "▲ worst per attribute   ★ best per attribute",
             ha="center", fontsize=8, color="gray")

    plt.savefig("bias_metrics_by_metric.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    metric_separate()
    print("done")