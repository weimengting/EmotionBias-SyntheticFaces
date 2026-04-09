import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# 1. Load CSV
# =====================================================
df = pd.read_csv("/Users/wmt/projects/bias/data/res/wan2.1/res.csv")

# =====================================================
# 2. Keep only face0
# =====================================================
df = df[df["face_name_align"].str.contains("face0", na=False)].copy()

# =====================================================
# 3. Parse GT attributes from path
#    .../selected_faces/<age>/<gender>/<race>/<file>
# =====================================================
def parse_gt_from_path(path):
    parts = str(path).split(os.sep)
    try:
        age = parts[-4]
        gender = parts[-3]
        race = parts[-2]
    except IndexError:
        age, gender, race = None, None, None
    return pd.Series([age, gender, race])

df[["gt_age_raw", "gt_gender_raw", "gt_race_raw"]] = \
    df["face_name_align"].apply(parse_gt_from_path)

# =====================================================
# 4. Normalize gender (GT & Pred)
# =====================================================
def norm_gender(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    if x in ["female", "f", "girl", "woman"]:
        return "female"
    if x in ["male", "m", "boy", "man"]:
        return "male"
    return None

df["gt_gender"] = df["gt_gender_raw"].apply(norm_gender)
df["gender_norm"] = df["gender"].apply(norm_gender)

# =====================================================
# 5. Align age bins (GT & Pred)
# =====================================================
def align_age(age):
    if pd.isna(age):
        return None
    age = str(age).strip()
    if age in ["0-2", "3-9", "0-9"]:
        return "0-9"
    if age == "10-19":
        return "10-19"
    if age in ["20-29", "30-39", "20-39"]:
        return "20-39"
    if age in ["40-49", "50-59", "40-59"]:
        return "40-59"
    if age in ["60-69", "70+", "60+"]:
        return "60+"
    return None

df["gt_age"] = df["gt_age_raw"].apply(align_age)
df["age_norm"] = df["age"].apply(align_age)

# =====================================================
# 6. Normalize race (GT & Pred) — 4 classes
# =====================================================
race_order_raw = [
    "White", "Black", "Latino_Hispanic",
    "East Asian", "Southeast Asian",
    "Indian", "Middle Eastern"
]

def parse_race_scores(x):
    x = str(x).replace("[", "").replace("]", "")
    return np.array([float(v) for v in x.split()])

def norm_race_4cls(r):
    if pd.isna(r):
        return None
    r = str(r).strip().lower().replace("_", " ")

    if r in ["east asian", "southeast asian", "asian"]:
        return "asian"
    if r in ["latino hispanic", "latino", "indian", "india"]:
        return "others"
    if r in ["white", "black"]:
        return r
    return None

def norm_race_pred_4cls(row):
    race = row["race"]
    scores_str = row.get("race_scores_fair", None)

    if pd.isna(race):
        return None

    race_str = str(race)

    # Middle Eastern → second highest score
    if race_str == "Middle Eastern" and scores_str is not None and not pd.isna(scores_str):
        scores = parse_race_scores(scores_str)
        idx = np.argsort(scores)[::-1][1]
        race_str = race_order_raw[idx]

    return norm_race_4cls(race_str)

df["gt_race_4cls"] = df["gt_race_raw"].apply(norm_race_4cls)
df["race_norm_4cls"] = df.apply(norm_race_pred_4cls, axis=1)

# =====================================================
# 7. Confusion matrices (row-normalized)
# =====================================================
def confusion_row_norm(gt, pred, order):
    cm = pd.crosstab(gt, pred, normalize="index")
    return cm.reindex(index=order, columns=order, fill_value=0.0)

gender_order = ["female", "male"]
race_order_4 = ["white", "black", "asian", "others"]
age_order = ["0-9", "10-19", "20-39", "40-59", "60+"]

cm_gender = confusion_row_norm(df["gt_gender"], df["gender_norm"], gender_order)
cm_race   = confusion_row_norm(df["gt_race_4cls"], df["race_norm_4cls"], race_order_4)
cm_age    = confusion_row_norm(df["gt_age"], df["age_norm"], age_order)

# =====================================================
# 8. Plot heatmaps (Gender → Race → Age)
# =====================================================
sns.set_theme(style="white", context="paper")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

heatmap_cfg = [
    {"cm": cm_gender, "title": "Gender (Wan2.1)", "cmap": "Purples"},
    {"cm": cm_race,   "title": "Race (Wan2.1)",   "cmap": "Blues"},
    {"cm": cm_age,    "title": "Age (Wan2.1)",    "cmap": "YlGnBu"}
]

for ax, cfg in zip(axes, heatmap_cfg):
    sns.heatmap(
        cfg["cm"],
        annot=True,
        fmt=".2f",
        cmap=cfg["cmap"],
        cbar=False,
        ax=ax,
        vmin=0,
        vmax=1,
        annot_kws={"size": 17}
    )

    # Axis labels
    ax.set_xlabel("Predicted label", fontsize=17)
    ax.set_ylabel("True label", fontsize=17)

    # === 关键修改：tick labels 首字母大写 ===
    ax.set_xticklabels([t.get_text().capitalize() for t in ax.get_xticklabels()])
    ax.set_yticklabels([t.get_text().capitalize() for t in ax.get_yticklabels()])

    ax.tick_params(axis="both", labelsize=17)
    ax.set_title(cfg["title"], fontsize=16)

plt.tight_layout()
plt.savefig("./confusion_matrices_wan2.1.png", dpi=300)
plt.show()


