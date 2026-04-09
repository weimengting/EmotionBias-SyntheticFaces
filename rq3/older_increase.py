import os
import json
import numpy as np
from scipy.stats import chi2_contingency

def test_claim_b_older_faces():
    # =========================
    # 1. Config
    # =========================
    base_dir = "/Users/wmt/projects/bias/write_into_json/age"
    models = [
        "flux", "proteus", "sd3", "sana",
        "hunyuan", "qwen", "kolors", "wan2.1"
    ]
    age_keys = ["0-9", "10-19", "20-39", "40-59", "60+"]
    target_emotions = ["angry", "disgusted"]

    # older 定义
    older_bins = ["40-59", "60+"]
    non_older_bins = ["0-9", "10-19", "20-39"]

    # =========================
    # 2. Helper: pool counts
    # =========================
    def pool_counts(emotion):
        pooled = np.zeros(len(age_keys), dtype=int)
        for model in models:
            file_path = os.path.join(base_dir, f"{model}_age.json")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            counts = np.array(
                [data[emotion].get(k, 0) for k in age_keys],
                dtype=int
            )
            pooled += counts
        return pooled

    # neutral pooled once
    pooled_neutral = pool_counts("neutral")

    print("=== Claim (b): Anger / Disgust increase the proportion of older faces ===\n")
    print("Age bins:", age_keys)
    print("Older bins:", older_bins)
    print("Neutral pooled counts:", pooled_neutral.tolist(), f"(total={pooled_neutral.sum()})\n")

    # =========================
    # 3. Test each target emotion
    # =========================
    all_results = {}

    for emotion in target_emotions:
        pooled_emotion = pool_counts(emotion)

        # ---------- 3.1 Full 2x5 age distribution test ----------
        table_full = np.array([
            pooled_emotion,
            pooled_neutral
        ])
        chi2_full, p_full, dof_full, expected_full = chi2_contingency(table_full)

        # ---------- 3.2 Older vs Non-older 2x2 test ----------
        age_to_idx = {k: i for i, k in enumerate(age_keys)}

        emo_older = sum(pooled_emotion[age_to_idx[k]] for k in older_bins)
        emo_non_older = sum(pooled_emotion[age_to_idx[k]] for k in non_older_bins)

        neu_older = sum(pooled_neutral[age_to_idx[k]] for k in older_bins)
        neu_non_older = sum(pooled_neutral[age_to_idx[k]] for k in non_older_bins)

        table_binary = np.array([
            [emo_older, emo_non_older],
            [neu_older, neu_non_older]
        ])
        chi2_bin, p_bin, dof_bin, expected_bin = chi2_contingency(table_binary)

        # ---------- 3.3 Older proportions ----------
        emo_total = pooled_emotion.sum()
        neu_total = pooled_neutral.sum()

        emo_older_prop = emo_older / emo_total
        neu_older_prop = neu_older / neu_total
        delta_older = emo_older_prop - neu_older_prop

        # ---------- 3.4 Print ----------
        print(f"--- Emotion: {emotion} ---")
        print("Pooled emotion counts: ", pooled_emotion.tolist(), f"(total={emo_total})")
        print("Pooled neutral counts: ", pooled_neutral.tolist(), f"(total={neu_total})")
        print()

        print("[Full age distribution: 2x5 chi-squared]")
        print(f"Chi-square statistic: {chi2_full:.4f}")
        print(f"Degrees of freedom:   {dof_full}")
        print(f"p-value:              {p_full:.4e}")
        print()

        print("[Older vs Non-older: 2x2 chi-squared]")
        print(f"{emotion} older / non-older:   [{emo_older}, {emo_non_older}]")
        print(f"neutral older / non-older: [{neu_older}, {neu_non_older}]")
        print(f"Chi-square statistic: {chi2_bin:.4f}")
        print(f"Degrees of freedom:   {dof_bin}")
        print(f"p-value:              {p_bin:.4e}")
        print()

        print("[Older proportion]")
        print(f"{emotion} older proportion:   {emo_older_prop:.4f} ({emo_older_prop*100:.2f}%)")
        print(f"neutral older proportion: {neu_older_prop:.4f} ({neu_older_prop*100:.2f}%)")
        print(f"Delta older proportion:   {delta_older:.4f} ({delta_older*100:.2f} percentage points)")
        print()

        if delta_older > 0:
            print(f"Conclusion: Older-face proportion is higher under {emotion} than under neutral.")
        else:
            print(f"Conclusion: Older-face proportion does NOT increase under {emotion}.")

        print("=" * 70)
        print()

        all_results[emotion] = {
            "pooled_emotion": pooled_emotion,
            "pooled_neutral": pooled_neutral,
            "full_age_test": {
                "chi2": chi2_full,
                "dof": dof_full,
                "p": p_full,
                "expected": expected_full
            },
            "older_binary_test": {
                "table": table_binary,
                "chi2": chi2_bin,
                "dof": dof_bin,
                "p": p_bin,
                "expected": expected_bin
            },
            "older_proportion": {
                "emotion": emo_older_prop,
                "neutral": neu_older_prop,
                "delta": delta_older
            }
        }

    return all_results


if __name__ == "__main__":
    results = test_claim_b_older_faces()