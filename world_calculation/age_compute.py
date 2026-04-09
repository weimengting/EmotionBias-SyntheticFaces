import pandas as pd

file_path = "/Users/wmt/projects/bias/data/world_dis/age_dist_POPULATION_SINGLE_AGE_BOTH_SEXES.xlsx"
df = pd.read_excel(file_path, skiprows=16)

# 只保留 World + 2023
df_2023 = df[(df["Region, subregion, country or area *"] == "World") & (df["Year"] == 2023)]

# 找到所有年龄列
age_cols = [col for col in df_2023.columns if str(col).isdigit() or "+" in str(col)]
print(age_cols)
# 取出这一行（只有一行）
row = df_2023.iloc[0][age_cols]
print(row)

age_population = []

for age, pop in row.items():
    if "+" in str(age):
        age_val = int(str(age).replace("+", ""))
    else:
        age_val = int(age)

    age_population.append((age_val, pop))
print(age_population)
df_age = pd.DataFrame(age_population, columns=["Age", "Population"])

def age_group(age):
    if age <= 9:
        return "0-9"
    elif age <= 19:
        return "10-19"
    elif age <= 39:
        return "20-39"
    elif age <= 59:
        return "40-59"
    else:
        return "60+"

df_age["AgeGroup"] = df_age["Age"].apply(age_group)

grouped = df_age.groupby("AgeGroup")["Population"].sum()

total = grouped.sum()
proportion = grouped / total

# 排序
order = ["0-9", "10-19", "20-39", "40-59", "60+"]
proportion = proportion.reindex(order)

# AgeGroup
# 0-9      0.165959
# 10-19    0.163733
# 20-39    0.297204
# 40-59    0.231504
# 60+      0.141601
# Name: Population, dtype: float64

print(proportion)

if __name__ == '__main__':
    print("done")