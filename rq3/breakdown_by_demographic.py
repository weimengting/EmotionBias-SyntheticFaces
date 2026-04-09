import numpy as np
import pandas as pd
import os

# =========================
# 1. 配置
# =========================
base_path = "/Users/wmt/projects/bias/data/corrected"

models = ["flux", "hunyuan", "kolors", "proteus", "qwen", "sana", "sd3", "wan2.1"]

emotions = ["neutral", "angry", "disgusted", "fearful", "happy", "sad", "surprised"]
emotions_no_neutral = ["angry", "disgusted", "fearful", "happy", "sad", "surprised"]

# =========================
# 2. 年龄分组函数
# =========================
young_bins  = ["0-2", "3-9", "10-19", "20-29", "30-39"]
middle_bins = ["40-49", "50-59"]
old_bins    = ["60-69", "70+"]

def age_group(age):
    if age in young_bins:  return "young"
    if age in middle_bins: return "middle-aged"
    if age in old_bins:    return "old"
    return np.nan

# =========================
# 3. 读取数据
# =========================
def load_full(model, emotion):
    path = os.path.join(base_path, model, f"{emotion}_res.csv")
    df = pd.read_csv(path)
    df = df[df["face_name_align"].str.contains("face0", na=False)]
    df["race_group"] = df["race_final"].apply(
        lambda r: "White" if r == "White" else "non-White"
    )
    df["age_group"] = df["age"].apply(age_group)
    df = df.dropna(subset=["age_group"])
    return df[["gender", "race_group", "age_group", "attractiveness"]]

# =========================
# 4. 计算某群体在某情绪下的低颜值比例（8模型均值）
# =========================
def low_att_prob(group_col, group_val, emotion):
    props = []
    for model in models:
        try:
            df = load_full(model, emotion)
            subset = df[df[group_col] == group_val]
            if len(subset) == 0:
                continue
            prop = (subset["attractiveness"] == 0).mean()
            props.append(prop)
        except FileNotFoundError:
            print(f"WARNING: {model}/{emotion}_res.csv not found, skipping")
    return np.mean(props)

# =========================
# 5. 计算 Delta
# =========================
print("=" * 70)
print("TASK (3): Breakdown by demographic group")
print("=" * 70)

header = f"{'':14}" + "".join(f"{e:>12}" for e in emotions_no_neutral)
divider = "-" * (14 + 12 * len(emotions_no_neutral))

# --- 性别 ---
print("\n[ Gender ] ΔP_low-att = P(low-att|g,e) - P(low-att|g,neutral)\n")
gender_groups  = ["Male", "Female"]
gender_neutral = {g: low_att_prob("gender", g, "neutral") for g in gender_groups}
gender_delta   = {}
for g in gender_groups:
    gender_delta[g] = {
        e: low_att_prob("gender", g, e) - gender_neutral[g]
        for e in emotions_no_neutral
    }

print(header)
print(divider)
for g in gender_groups:
    row = f"{g:14}" + "".join(f"{gender_delta[g][e]*100:>+11.1f}%" for e in emotions_no_neutral)
    print(row)
print(f"\n  Neutral baseline: Male={gender_neutral['Male']*100:.1f}%, Female={gender_neutral['Female']*100:.1f}%")

# --- 种族 ---
print("\n[ Race ] ΔP_low-att = P(low-att|r,e) - P(low-att|r,neutral)\n")
race_groups  = ["White", "non-White"]
race_neutral = {r: low_att_prob("race_group", r, "neutral") for r in race_groups}
race_delta   = {}
for r in race_groups:
    race_delta[r] = {
        e: low_att_prob("race_group", r, e) - race_neutral[r]
        for e in emotions_no_neutral
    }

print(header)
print(divider)
for r in race_groups:
    row = f"{r:14}" + "".join(f"{race_delta[r][e]*100:>+11.1f}%" for e in emotions_no_neutral)
    print(row)
print(f"\n  Neutral baseline: White={race_neutral['White']*100:.1f}%, non-White={race_neutral['non-White']*100:.1f}%")

# --- 年龄 ---
print("\n[ Age ] ΔP_low-att = P(low-att|a,e) - P(low-att|a,neutral)\n")
age_groups  = ["young", "middle-aged", "old"]
age_neutral = {a: low_att_prob("age_group", a, "neutral") for a in age_groups}
age_delta   = {}
for a in age_groups:
    age_delta[a] = {
        e: low_att_prob("age_group", a, e) - age_neutral[a]
        for e in emotions_no_neutral
    }

print(header)
print(divider)
for a in age_groups:
    row = f"{a:14}" + "".join(f"{age_delta[a][e]*100:>+11.1f}%" for e in emotions_no_neutral)
    print(row)
print(f"\n  Neutral baseline:")
for a in age_groups:
    print(f"    {a}: {age_neutral[a]*100:.1f}%")

# =========================
# 6. 复合刻板印象检验
# =========================
print("\n" + "=" * 70)
print("Compound stereotyping check (Anger):")
print(f"  Male        Δ = {gender_delta['Male']['angry']*100:+.1f}%")
print(f"  Female      Δ = {gender_delta['Female']['angry']*100:+.1f}%")
print(f"  White       Δ = {race_delta['White']['angry']*100:+.1f}%")
print(f"  non-White   Δ = {race_delta['non-White']['angry']*100:+.1f}%")
print(f"  young       Δ = {age_delta['young']['angry']*100:+.1f}%")
print(f"  middle-aged Δ = {age_delta['middle-aged']['angry']*100:+.1f}%")
print(f"  old         Δ = {age_delta['old']['angry']*100:+.1f}%")
print("=" * 70)

if __name__ == '__main__':
    print("done")