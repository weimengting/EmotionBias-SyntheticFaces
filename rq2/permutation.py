import numpy as np
import itertools
import random

def permutation_test(values, western_idx, B=10000, seed=42):
    """
    values: list of 8 TVD values
    western_idx: indices of western models (length=4)
    B: number of permutations (如果想精确，可以用 exact=True)
    """

    np.random.seed(seed)
    random.seed(seed)

    values = np.array(values)

    # ===== 原始分组 =====
    western = values[western_idx]
    chinese = np.delete(values, western_idx)

    obs_diff = abs(western.mean() - chinese.mean())

    # ===== 所有可能组合（精确解）=====
    all_indices = list(range(len(values)))
    all_combinations = list(itertools.combinations(all_indices, 4))

    perm_diffs = []

    for comb in all_combinations:
        group_A = values[list(comb)]
        group_B = np.delete(values, list(comb))

        diff = abs(group_A.mean() - group_B.mean())
        perm_diffs.append(diff)

    perm_diffs = np.array(perm_diffs)

    # ===== p-value =====
    p_value = np.mean(perm_diffs >= obs_diff)

    return obs_diff, p_value, perm_diffs

models = ["flux", "proteus", "sd3", "sana", "hunyuan", "qwen", "kolors", "wan2.1"]

# 前4个是 Western
western_idx = [0, 1, 2, 3]
gender_tvd = [0.14, 0.47, 0.33, 0.45, 0.32, 0.37, 0.04, 0.03]

obs, p, _ = permutation_test(gender_tvd, western_idx)

print("Gender:")
print("Observed diff:", round(obs, 3))
print("p-value:", round(p, 3))

race_tvd = [0.24, 0.81, 0.23, 0.72, 0.64, 0.23, 0.74, 0.39]

obs, p, _ = permutation_test(race_tvd, western_idx)

print("\nRace:")
print("Observed diff:", round(obs, 3))
print("p-value:", round(p, 3))

age_tvd = [0.40, 0.62, 0.67, 0.54, 0.53, 0.52, 0.56, 0.67]

obs, p, _ = permutation_test(age_tvd, western_idx)

print("\nAge:")
print("Observed diff:", round(obs, 3))
print("p-value:", round(p, 3))

# Gender:
# Observed diff: 0.157
# p-value: 0.2
#
# Race:
# Observed diff: 0.0
# p-value: 1.0
#
# Age:
# Observed diff: 0.013
# p-value: 0.886


if __name__ == '__main__':
    print("done")