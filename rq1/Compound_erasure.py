import numpy as np
import pandas as pd
from itertools import product

# =========================
# 1. 配置
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

real_world = {
    "White":  0.1307,
    "Black":  0.1689,
    "Asian":  0.3199,
    "Others": 0.3805
}

real_world_age = {
    "0-9":   0.165959,
    "10-19": 0.163733,
    "20-39": 0.297204,
    "40-59": 0.231504,
    "60+":   0.141601
}

real_world_gender = {
    "Male":   0.5029,
    "Female": 0.4971
}

# =========================
# 2. Combo 顺序（2×4×3 = 24）
# =========================
age_order    = ["young", "middle-aged", "old"]
gender_order = ["male", "female"]
race_order   = ["white", "black", "asian", "others"]

full_combo_order = [
    f"{a}-{g}-{r}"
    for a, g, r in product(age_order, gender_order, race_order)
]
assert len(full_combo_order) == 24

# =========================
# 3. 数据处理辅助函数
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

def process_one_model(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["face_name_align"].str.contains("face0", na=False)]
    df["race_new"]  = df.apply(process_race, axis=1)
    df["age_group"] = df["age"].apply(age_group)
    df = df.dropna(subset=["age_group"])
    df["gender"]    = df["gender"].str.lower()
    df["combo"]     = df["age_group"] + "-" + df["gender"] + "-" + df["race_new"]

    obs         = df["combo"].value_counts()
    combo_count = obs.reindex(full_combo_order).fillna(0).astype(int)
    combo_prob  = combo_count / combo_count.sum()
    return combo_prob.values  # shape: (24,)

# =========================
# 4. 读取所有模型
# =========================
results = {}
for model_name, path in model_files.items():
    results[model_name] = process_one_model(path)
    print(f"Loaded: {model_name}")

# =========================
# 任务(1)：近零交叉单元格
# =========================
THRESHOLD = 0.01
NUM_CELLS = 24

print("\n" + "=" * 60)
print("TASK (1): Near-zero intersectional cells")
print("=" * 60)

near_zero_counts = {}
for model_name, probs in results.items():
    count = int(np.sum(probs < THRESHOLD))
    near_zero_counts[model_name] = count
    print(f"  {model_name}: {count}/{NUM_CELLS} near-zero cells")

counts     = np.array(list(near_zero_counts.values()))
mean_count = counts.mean()
std_count  = counts.std(ddof=1)
min_count  = counts.min()
max_count  = counts.max()
mean_pct   = mean_count / NUM_CELLS * 100

print(f"\nResult:")
print(
    f'On average, {mean_count:.1f} ± {std_count:.1f} of {NUM_CELLS} '
    f'intersectional cells ({mean_pct:.0f}%) had near-zero representation '
    f'(P < 0.01) across all models.'
)
print(f"[Range: {min_count}–{max_count} cells across {len(counts)} models]")

# =========================
# 任务(2)：young × female × Black
# =========================
print("\n" + "=" * 60)
print("TASK (2): Compound erasure of young × female × Black")
print("=" * 60)

# 期望概率
P_young  = real_world_age["0-9"] + real_world_age["10-19"] + real_world_age["20-39"]
P_female = real_world_gender["Female"]
P_black  = real_world["Black"]
P_exp    = P_young * P_female * P_black

print(f"\nExpected probability (under unbiased generator):")
print(f"  P(young)  = {P_young:.4f} ({P_young*100:.1f}%)")
print(f"  P(female) = {P_female:.4f} ({P_female*100:.1f}%)")
print(f"  P(Black)  = {P_black:.4f} ({P_black*100:.1f}%)")
print(f"  P_exp     = {P_exp:.4f} ({P_exp*100:.2f}%)")

# 观测概率
target_idx = full_combo_order.index("young-female-black")

observed_probs = {}
for model_name, prob_array in results.items():
    observed_probs[model_name] = prob_array[target_idx]

obs_values = np.array(list(observed_probs.values()))
obs_mean   = obs_values.mean()

print(f"\nObserved P(young-female-black) per model:")
for model, p in observed_probs.items():
    print(f"  {model}: {p:.4f} ({p*100:.2f}%)")

# Bootstrap 95% CI
np.random.seed(42)
n_bootstrap = 10000
boot_means  = [
    np.random.choice(obs_values, size=len(obs_values), replace=True).mean()
    for _ in range(n_bootstrap)
]
ci_lower = np.percentile(boot_means, 2.5)
ci_upper = np.percentile(boot_means, 97.5)

# 压制因子
suppression_factor = obs_mean / P_exp

print(f"\nResult:")
print(
    f'The young × female × Black intersection was generated '
    f'at {obs_mean*100:.1f}% probability across models '
    f'(expected: {P_exp*100:.1f}% under an unbiased generator; '
    f'suppression factor: {suppression_factor:.2f}×).'
)
print(f"95% Bootstrap CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
print("=" * 60)

# =========================
# 任务(3)：Western vs Chinese 交叉分布比较
# =========================
from scipy.spatial.distance import jensenshannon

print("\n" + "=" * 60)
print("TASK (3): Formal Western vs. Chinese intersectional comparison")
print("=" * 60)

# 分组
western_models = ["FLUX", "Proteus", "SANA", "SD3"]
chinese_models = ["Hunyuan", "Kolors", "Qwen", "Wan2.1"]

# =========================
# 第一步：计算组内平均分布
# =========================
P_W = np.mean([results[m] for m in western_models], axis=0)  # shape: (24,)
P_C = np.mean([results[m] for m in chinese_models], axis=0)  # shape: (24,)

print("\nAggregate Western distribution (top 5 combos):")
top5_w = np.argsort(P_W)[::-1][:5]
for i in top5_w:
    print(f"  {full_combo_order[i]}: {P_W[i]:.4f} ({P_W[i]*100:.1f}%)")

print("\nAggregate Chinese distribution (top 5 combos):")
top5_c = np.argsort(P_C)[::-1][:5]
for i in top5_c:
    print(f"  {full_combo_order[i]}: {P_C[i]:.4f} ({P_C[i]*100:.1f}%)")

# =========================
# 第二步：组间 JS 散度
# =========================
# scipy的jensenshannon返回的是JS距离（JS散度的平方根），需要平方还原
js_between = jensenshannon(P_W, P_C) ** 2

print(f"\nBetween-group JS divergence:")
print(f"  D_JS(P_W, P_C) = {js_between:.4f}")

# =========================
# 第三步：组内平均 JS 散度
# =========================
from itertools import combinations

def mean_pairwise_js(model_list):
    pairs = list(combinations(model_list, 2))  # C(4,2) = 6 pairs
    js_values = []
    for m1, m2 in pairs:
        js = jensenshannon(results[m1], results[m2]) ** 2
        js_values.append(js)
        print(f"    {m1} vs {m2}: {js:.4f}")
    return np.mean(js_values), js_values

print(f"\nWithin-group JS divergence (Western, {len(list(combinations(western_models,2)))} pairs):")
js_within_W, js_values_W = mean_pairwise_js(western_models)
print(f"  Mean D_JS within Western = {js_within_W:.4f}")

print(f"\nWithin-group JS divergence (Chinese, {len(list(combinations(chinese_models,2)))} pairs):")
js_within_C, js_values_C = mean_pairwise_js(chinese_models)
print(f"  Mean D_JS within Chinese = {js_within_C:.4f}")

# =========================
# 第四步：输出报告
# =========================
print("\n" + "=" * 60)
print("RESULT:")
print(
    f'The JS divergence between the aggregate Western and Chinese '
    f'intersectional distributions (D_JS = {js_between:.4f}) was comparable '
    f'to the within-group mean JS divergence among Western models '
    f'({js_within_W:.4f}) and Chinese models ({js_within_C:.4f}), '
    f'indicating that regional origin explains less variance than '
    f'individual model differences.'
)
print("=" * 60)



# ============================================================
# TASK (2): Compound erasure of young × female × Black
# ============================================================
#
# Expected probability (under unbiased generator):
#   P(young)  = 0.6269 (62.7%)
#   P(female) = 0.4971 (49.7%)
#   P(Black)  = 0.1689 (16.9%)
#   P_exp     = 0.0526 (5.26%)
#
# Observed P(young-female-black) per model:
#   Hunyuan: 0.0000 (0.00%)
#   Kolors: 0.0000 (0.00%)
#   Qwen: 0.0170 (1.70%)
#   Wan2.1: 0.0640 (6.40%)
#   FLUX: 0.0190 (1.90%)
#   Proteus: 0.0000 (0.00%)
#   SANA: 0.0000 (0.00%)
#   SD3: 0.0060 (0.60%)
#
# Result:
# The young × female × Black intersection was generated at 1.3% probability across models (expected: 5.3% under an unbiased generator; suppression factor: 0.25×).
# 95% Bootstrap CI: [0.21%, 2.93%]
# ============================================================


#============================================================
# TASK (3): Formal Western vs. Chinese intersectional comparison
# ============================================================
#
# Aggregate Western distribution (top 5 combos):
#   young-male-white: 0.2650 (26.5%)
#   young-female-white: 0.2620 (26.2%)
#   young-male-asian: 0.1535 (15.3%)
#   young-male-others: 0.0788 (7.9%)
#   young-female-asian: 0.0668 (6.7%)
#
# Aggregate Chinese distribution (top 5 combos):
#   young-female-asian: 0.2320 (23.2%)
#   young-male-white: 0.2208 (22.1%)
#   young-female-white: 0.1450 (14.5%)
#   young-male-others: 0.1328 (13.3%)
#   young-male-black: 0.0748 (7.5%)
#
# Between-group JS divergence:
#   D_JS(P_W, P_C) = 0.0634
#
# Within-group JS divergence (Western, 6 pairs):
#     FLUX vs Proteus: 0.3747
#     FLUX vs SANA: 0.3259
#     FLUX vs SD3: 0.1224
#     Proteus vs SANA: 0.5356
#     Proteus vs SD3: 0.4859
#     SANA vs SD3: 0.4874
#   Mean D_JS within Western = 0.3887
#
# Within-group JS divergence (Chinese, 6 pairs):
#     Hunyuan vs Kolors: 0.5972
#     Hunyuan vs Qwen: 0.4673
#     Hunyuan vs Wan2.1: 0.4306
#     Kolors vs Qwen: 0.3723
#     Kolors vs Wan2.1: 0.1378
#     Qwen vs Wan2.1: 0.2035
#   Mean D_JS within Chinese = 0.3681
#
# ============================================================
# RESULT:
# The JS divergence between the aggregate Western and Chinese intersectional distributions (D_JS = 0.0634) was comparable to the within-group mean JS divergence among Western models (0.3887) and Chinese models (0.3681), indicating that regional origin explains less variance than individual model differences.
# ============================================================

if __name__ == '__main__':
    print("done")