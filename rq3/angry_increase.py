import os
import json
import numpy as np
from scipy.stats import chi2_contingency

def test_claim_a_anger_white_representation():
    # =========================
    # 1. Config
    # =========================
    base_dir = "/Users/wmt/projects/bias/write_into_json/race"
    models = [
        "flux", "proteus", "sd3", "sana",
        "hunyuan", "qwen", "kolors", "wan2.1"
    ]
    race_keys = ["White", "Black", "Asian", "Others"]

    # =========================
    # 2. Pool counts across all models
    # =========================
    pooled_neutral = np.zeros(len(race_keys), dtype=int)
    pooled_angry = np.zeros(len(race_keys), dtype=int)

    for model in models:
        file_path = os.path.join(base_dir, f"{model}_race.json")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        neutral_counts = np.array(
            [data["neutral"].get(k, 0) for k in race_keys],
            dtype=int
        )
        angry_counts = np.array(
            [data["angry"].get(k, 0) for k in race_keys],
            dtype=int
        )

        pooled_neutral += neutral_counts
        pooled_angry += angry_counts

    # =========================
    # 3. Build 2 x K contingency table
    # =========================
    contingency_table = np.array([
        pooled_angry,
        pooled_neutral
    ])

    # =========================
    # 4. Chi-squared test
    # =========================
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # =========================
    # 5. White proportion change
    # =========================
    white_idx = race_keys.index("White")

    angry_total = pooled_angry.sum()
    neutral_total = pooled_neutral.sum()

    white_prop_angry = pooled_angry[white_idx] / angry_total
    white_prop_neutral = pooled_neutral[white_idx] / neutral_total
    delta_white = white_prop_angry - white_prop_neutral

    # =========================
    # 6. Print results
    # =========================
    print("=== Claim (a): Anger increases White representation ===\n")

    print("Race categories:", race_keys)
    print("Pooled angry counts:  ", pooled_angry.tolist(), f"(total={angry_total})")
    print("Pooled neutral counts:", pooled_neutral.tolist(), f"(total={neutral_total})")
    print()

    print(f"White proportion under angry:   {white_prop_angry:.4f} ({white_prop_angry*100:.2f}%)")
    print(f"White proportion under neutral: {white_prop_neutral:.4f} ({white_prop_neutral*100:.2f}%)")
    print(f"Delta White proportion:         {delta_white:.4f} ({delta_white*100:.2f} percentage points)")
    print()

    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom:   {dof}")
    print(f"p-value:              {p:.4e}")
    print()

    if delta_white > 0:
        print("Conclusion: White representation is higher under anger than under neutral.")
    else:
        print("Conclusion: White representation does NOT increase under anger.")

    if p < 0.05:
        print("The overall race distribution shift is statistically significant at p < 0.05.")
    else:
        print("The overall race distribution shift is NOT statistically significant at p < 0.05.")

    return {
        "pooled_angry": pooled_angry,
        "pooled_neutral": pooled_neutral,
        "white_prop_angry": white_prop_angry,
        "white_prop_neutral": white_prop_neutral,
        "delta_white": delta_white,
        "chi2": chi2,
        "dof": dof,
        "p": p,
        "expected": expected
    }


if __name__ == "__main__":
    results = test_claim_a_anger_white_representation()