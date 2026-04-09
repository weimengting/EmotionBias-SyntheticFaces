import pandas as pd
import numpy as np
from itertools import product
import os
from scipy.spatial.distance import jensenshannon

# =====================================================
# 1. Model paths
# =====================================================
base_dir = "/Users/wmt/projects/bias/data/corrected"

models = [
    "flux",
    "proteus",
    "sd3",
    "sana",
    "hunyuan",
    "qwen",
    "kolors",
    "wan2.1",
]

expression_files = {
    "happy": "happy_res.csv",
    "sad": "sad_res.csv",
    "angry": "angry_res.csv",
    "surprised": "surprised_res.csv",
    "fear": "fearful_res.csv",
    "disgusted": "disgusted_res.csv",
}

neutral_filename = "neutral_res.csv"

# =====================================================
# 2. Shared processing logic
# =====================================================
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

young_bins = ["0-2", "3-9", "10-19", "20-29", "30-39"]
middle_bins = ["40-49", "50-59"]
old_bins = ["60-69", "70+"]

def age_group(age):
    if age in young_bins:
        return "young"
    if age in middle_bins:
        return "middle-aged"
    if age in old_bins:
        return "old"
    return np.nan

age_order = ["young", "middle-aged", "old"]
gender_order = ["Male", "Female"]
race_order = ["white", "black", "asian", "others"]

full_support = [
    f"{a}-{g}-{r}"
    for a, g, r in product(age_order, gender_order, race_order)
]

# =====================================================
# 3. Joint distribution
# =====================================================
def joint_distribution(csv_path):
    df = pd.read_csv(csv_path)

    # face0 only
    df = df[df["face_name_align"].str.contains("face0", na=False)].copy()

    df["race_new"] = df.apply(process_race, axis=1)
    df["age_group"] = df["age"].apply(age_group)
    df["gender"] = df["gender"].str.strip().str.capitalize()
    df = df.dropna(subset=["age_group"])

    combo = (
        df["age_group"] + "-" +
        df["gender"] + "-" +
        df["race_new"]
    )

    counts = combo.value_counts().reindex(full_support).fillna(0).astype(float).values

    total = counts.sum()
    if total == 0:
        raise ValueError(f"No valid samples found in {csv_path}")

    probs = counts / total
    return probs

# =====================================================
# 4. KL & JS
# =====================================================
def kl_divergence(P, Q, eps=1e-12):
    P = np.asarray(P, dtype=float) + eps
    Q = np.asarray(Q, dtype=float) + eps
    P /= P.sum()
    Q /= Q.sum()
    return float(np.sum(P * np.log(P / Q)))

def js_divergence(P, Q):
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    return float(jensenshannon(P, Q, base=2) ** 2)

# =====================================================
# 4.5 Bonferroni correction
# =====================================================
num_tests = len(models) * len(expression_files)
alpha = 0.05
alpha_star = alpha / num_tests

print(f"[Info] Bonferroni correction: alpha* = {alpha_star:.6e} (N={num_tests})")

# =====================================================
# 5. Compute for all models
# =====================================================
all_records = []

for model in models:
    model_dir = os.path.join(base_dir, model)
    neutral_path = os.path.join(model_dir, neutral_filename)

    if not os.path.exists(neutral_path):
        print(f"[Warning] Neutral file not found for model {model}: {neutral_path}")
        continue

    try:
        P_neutral = joint_distribution(neutral_path)
    except Exception as e:
        print(f"[Warning] Failed to process neutral file for {model}: {e}")
        continue

    for expr, filename in expression_files.items():
        expr_path = os.path.join(model_dir, filename)

        if not os.path.exists(expr_path):
            print(f"[Warning] Expression file not found: {expr_path}")
            continue

        try:
            P_expr = joint_distribution(expr_path)

            kl = kl_divergence(P_expr, P_neutral)
            js = js_divergence(P_expr, P_neutral)

            # ===============================
            # ✅ Bonferroni significance
            # ===============================
            significant = js > alpha_star

            all_records.append({
                "model": model,
                "expression": expr,
                "KL(expr || neutral)": kl,
                "JS(expr, neutral)": js,
                "alpha_star": alpha_star,
                "significant": significant,
                "significant_mark": "*" if significant else ""
            })

        except Exception as e:
            print(f"[Warning] Failed on {model} / {expr}: {e}")

# =====================================================
# 6. Results table
# =====================================================
df_results = pd.DataFrame(all_records)

if len(df_results) == 0:
    print("No results were computed.")
else:
    df_results = df_results.sort_values(["model", "KL(expr || neutral)"])
    print(df_results)

    os.makedirs("./results", exist_ok=True)

    # ===============================
    # ✅ 主结果（带显著性）
    # ===============================
    df_results.to_csv(
        "./results/intersectional_kl_js_vs_neutral_all_models_with_significance.csv",
        index=False
    )

    # ===============================
    # ✅ 每个模型单独保存
    # ===============================
    for model in df_results["model"].unique():
        df_model = df_results[df_results["model"] == model].sort_values("KL(expr || neutral)")
        df_model.to_csv(
            f"./results/{model}_intersectional_with_significance.csv",
            index=False
        )

    # ===============================
    # ✅ 保存metadata（论文直接用）
    # ===============================
    with open("./results/statistical_test_info.txt", "w") as f:
        f.write("Statistical testing information\n")
        f.write("--------------------------------\n")
        f.write(f"Number of tests (N): {num_tests}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Bonferroni corrected alpha*: {alpha_star:.6e}\n")
        f.write("\nCriterion:\n")
        f.write("JS divergence > alpha* is considered statistically significant.\n")



if __name__ == '__main__':
    print("done")