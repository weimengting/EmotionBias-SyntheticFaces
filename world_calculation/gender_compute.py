import pandas as pd

file_path = "/Users/wmt/projects/bias/data/world_dis/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx"
df = pd.read_excel(file_path, skiprows=16)

# 过滤
df_2023 = df[
    (df["Region, subregion, country or area *"] == "World") &
    (df["Year"] == 2023)
]

# 自动找列
male_col = [col for col in df.columns if "male" in col.lower() and "population" in col.lower()][0]
female_col = [col for col in df.columns if "female" in col.lower() and "population" in col.lower()][0]

male = df_2023[male_col].values[0]
female = df_2023[female_col].values[0]

ratio = male / female
print(ratio)
# ===== distribution =====
total = male + female
p_male = male / total
p_female = female / total

print("Distribution:")
print({
    "male": p_male,
    "female": p_female
})
# {'male': 0.5029022778431522, 'female': 0.4970977221568478}


if __name__ == '__main__':
    print("done")