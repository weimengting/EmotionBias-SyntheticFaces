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

# ===== 定义 mapping =====
mapping = {
    "Asian": ["Eastern Asia", "South-Eastern Asia"],
    "Indian": ["India"],
    "Black": ["Sub-Saharan Africa"],
    "White": ["Europe and Northern America"],
    "Latino": ["Latin America and the Caribbean"]
}

# ===== 计算每个group人口 =====
result = {}

for group, regions in mapping.items():
    subset = df[df[region_col].isin(regions)]
    total = subset["TotalPop"].sum()
    result[group] = total

# ===== 转成 distribution =====
total_population = sum(result.values())
distribution = {k: v / total_population for k, v in result.items()}

# ===== 输出 =====
print("Population:")
for k, v in result.items():
    print(k, v)

print("\nDistribution:")
for k, v in distribution.items():
    print(f"{k}: {v:.4f}")


# Distribution:
# Asian: 0.3156
# Indian: 0.1931
# Black: 0.1628
# White: 0.1515
# Latino: 0.1770

if __name__ == '__main__':
    print("done")