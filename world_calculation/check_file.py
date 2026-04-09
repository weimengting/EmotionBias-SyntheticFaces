import pandas as pd

file_path = "/Users/wmt/projects/bias/data/world_dis/age_dist_POPULATION_SINGLE_AGE_BOTH_SEXES.xlsx"
df = pd.read_excel(file_path, skiprows=16)

# ===== 只取2023 =====
df = df[df["Year"] == 2023]

# ===== 找所有年龄列 =====
age_cols = [col for col in df.columns if str(col).isdigit() or "+" in str(col)]

# ===== 计算每一行的总人口 =====
df["TotalPop"] = df[age_cols].sum(axis=1)

region_col = "Region, subregion, country or area *"

# =========================
# 1. 直接地区映射（无需拆分）
# =========================
region_mapping = {
    "Asian": ["Eastern Asia", "South-Eastern Asia"],
    "Indian": ["India"],
    "Black": ["Sub-Saharan Africa"],
    "Latino": ["Latin America and the Caribbean"],
}

result = {}
for group, regions in region_mapping.items():
    subset = df[df[region_col].isin(regions)]
    result[group] = subset["TotalPop"].sum()

# =========================
# 2. White = Europe + 美国白人 + 加拿大白人
# =========================

# 欧洲总人口
europe_pop = df[df[region_col] == "Europe"]["TotalPop"].values[0]

# 美国人口 × 白人比例
us_pop = df[df[region_col] == "United States of America"]["TotalPop"].values[0]
us_race = {
    "White":  0.578,
    "Latino": 0.187,
    "Black":  0.121,
    "Asian":  0.059,
    "Indian": 0.013,
}

# 加拿大人口 × 白人比例
ca_pop = df[df[region_col] == "Canada"]["TotalPop"].values[0]
ca_race = {
    "White":  0.698,
    "Asian":  0.177,
    "Black":  0.035,
    "Latino": 0.015,
    "Indian": 0.071,
}

result["White"] = europe_pop + us_pop * us_race["White"] + ca_pop * ca_race["White"]

# =========================
# 3. 把美加的其他族裔也加进去
# =========================
for group in ["Black", "Latino", "Asian", "Indian"]:
    result[group] += us_pop * us_race.get(group, 0)
    result[group] += ca_pop * ca_race.get(group, 0)

# =========================
# 4. 转成分布
# =========================
total_population = sum(result.values())
distribution = {k: v / total_population for k, v in result.items()}

# =========================
# 5. 输出
# =========================
print("Population (thousands):")
for k, v in result.items():
    print(f"  {k}: {v:,.0f}")

print("\nDistribution:")
for k, v in distribution.items():
    print(f"  {k}: {v:.4f} ({v*100:.1f}%)")

print(f"\nTotal: {total_population:,.0f}")
print(f"Sum of distribution: {sum(distribution.values()):.4f}")

# Distribution:
#   Asian: 0.3199 (32.0%)
#   Indian: 0.1945 (19.4%)
#   Black: 0.1689 (16.9%)
#   Latino: 0.1860 (18.6%)
#   White: 0.1307 (13.1%)

if __name__ == '__main__':
    print("done")