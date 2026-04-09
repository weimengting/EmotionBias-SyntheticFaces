def compute_near_zero_intersectional():
    import numpy as np
    import pandas as pd
    from itertools import product

    # =========================
    # 1. Load data - 补全8个模型
    # =========================
    model_files = {
        "Hunyuan": "/Users/wmt/projects/bias/data/corrected/hunyuan/neutral_res.csv",
        "Kolors":  "/Users/wmt/projects/bias/data/corrected/kolors/neutral_res.csv",
        "Qwen":    "/Users/wmt/projects/bias/data/corrected/qwen/neutral_res.csv",
        "Wan2.1":  "/Users/wmt/projects/bias/data/corrected/wan2.1/neutral_res.csv",
        "FLUX": "/Users/wmt/projects/bias/data/corrected/flux/neutral_res.csv",
        "Proteus": "/Users/wmt/projects/bias/data/corrected/proteus/neutral_res.csv",
        "SANA": "/Users/wmt/projects/bias/data/corrected/sana/neutral_res.csv",
        "SD3": "/Users/wmt/projects/bias/data/corrected/sd3/neutral_res.csv",
    }

    # =========================
    # 2. Fixed semantic order (2×4×3 = 24 cells)
    # =========================
    age_order    = ["young", "middle-aged", "old"]
    gender_order = ["male", "female"]
    race_order   = ["white", "black", "asian", "others"]

    full_combo_order = [
        f"{a}-{g}-{r}"
        for a, g, r in product(age_order, gender_order, race_order)
    ]
    # 验证：应该是24
    assert len(full_combo_order) == 24, f"Expected 24 cells, got {len(full_combo_order)}"

    # =========================
    # 3. Race + age helpers
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
    # 4. Process one model → return prob array
    # =========================
    def process_one_model(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["face_name_align"].str.contains("face0", na=False)]
        df["race_new"]   = df.apply(process_race, axis=1)
        df["age_group"]  = df["age"].apply(age_group)
        df = df.dropna(subset=["age_group"])
        df["gender"]     = df["gender"].str.lower()
        df["combo"]      = df["age_group"] + "-" + df["gender"] + "-" + df["race_new"]

        obs         = df["combo"].value_counts()
        combo_count = obs.reindex(full_combo_order).fillna(0).astype(int)
        combo_prob  = combo_count / combo_count.sum()
        return combo_prob.values  # shape: (24,)

    # =========================
    # 5. Compute near-zero stats
    # =========================
    THRESHOLD   = 0.01   # P(d) < 0.01
    NUM_CELLS   = 24

    near_zero_counts = {}   # model_name → int (number of near-zero cells)

    for model_name, path in model_files.items():
        probs = process_one_model(path)
        count = int(np.sum(probs < THRESHOLD))
        near_zero_counts[model_name] = count
        print(f"  {model_name}: {count}/{NUM_CELLS} near-zero cells")

    # 在 step 5 之后加入以下代码

    # =========================
    # 6.5 Least represented cells
    # =========================
    all_probs = {}  # model_name → prob array

    for model_name, path in model_files.items():
        probs = process_one_model(path)
        all_probs[model_name] = probs

    # 每个 cell 在所有模型上的平均概率
    mean_probs = np.mean(
        np.stack(list(all_probs.values()), axis=0), axis=0
    )  # shape: (24,)

    # 按平均概率排序，取最低的几个
    sorted_idx = np.argsort(mean_probs)
    print("\nLeast represented intersectional cells (averaged across 8 models):")
    print(f"{'Cell':<30} {'Mean P':>8}  " + "  ".join([f"{m[:6]:>7}" for m in model_files]))
    for idx in sorted_idx[:8]:
        cell = full_combo_order[idx]
        mean_p = mean_probs[idx]
        per_model = "  ".join([f"{all_probs[m][idx]:.4f}" for m in model_files])
        print(f"{cell:<30} {mean_p:.4f}    {per_model}")

    # =========================
    # 6. Summarize across models
    # =========================
    counts = np.array(list(near_zero_counts.values()))

    mean_count = counts.mean()
    std_count  = counts.std(ddof=1)   # ± 用标准差
    min_count  = counts.min()
    max_count  = counts.max()
    mean_pct   = mean_count / NUM_CELLS * 100

    # 格式：X ± Y (range: min–max)
    print("\n" + "="*65)
    print("RESULT:")
    print(
        f'On average, {mean_count:.1f} ± {std_count:.1f} of {NUM_CELLS} '
        f'intersectional cells ({mean_pct:.0f}%) had near-zero representation '
        f'(P < 0.01) across all models.'
    )
    print(f"[Range: {min_count}–{max_count} cells across {len(counts)} models]")
    print("="*65)

    return near_zero_counts

# Hunyuan: 18/24 near-zero cells
#   Kolors: 18/24 near-zero cells
#   Qwen: 13/24 near-zero cells
#   Wan2.1: 15/24 near-zero cells
#   FLUX: 11/24 near-zero cells
#   Proteus: 20/24 near-zero cells
#   SANA: 20/24 near-zero cells
#   SD3: 17/24 near-zero cells
#
# =================================================================
# RESULT:
# On average, 16.5 ± 3.3 of 24 intersectional cells (69%) had near-zero representation (P < 0.01) across all models.
# [Range: 11–20 cells across 8 models]
# =================================================================

if __name__ == '__main__':
    compute_near_zero_intersectional()