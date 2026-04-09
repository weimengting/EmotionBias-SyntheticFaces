import os
import json
import numpy as np

def test_claim_c_happiness_smallest_shift():
    # =========================
    # 1. Config
    # =========================
    base_dirs = {
        "age": "/Users/wmt/projects/bias/write_into_json/age",
        "gender": "/Users/wmt/projects/bias/write_into_json/gender",
        "race": "/Users/wmt/projects/bias/write_into_json/race",
        "attractiveness": "/Users/wmt/projects/bias/write_into_json/attractiveness"
    }

    category_keys = {
        "age": ["0-9", "10-19", "20-39", "40-59", "60+"],
        "gender": ["Male", "Female"],
        "race": ["White", "Black", "Asian", "Others"],
        "attractiveness": ["0", "1", "2"]
    }

    models = [
        "flux", "proteus", "sd3", "sana",
        "hunyuan", "qwen", "kolors", "wan2.1"
    ]

    emotions = ["happy", "surprised", "sad", "fearful", "angry", "disgusted"]

    # =========================
    # 2. TVD function
    # =========================
    def compute_tvd(p, q):
        return 0.5 * np.sum(np.abs(p - q))

    # =========================
    # 3. Collect TVDs for each emotion
    # =========================
    emotion_tvds = {e: [] for e in emotions}

    for attr, base_dir in base_dirs.items():
        keys = category_keys[attr]

        for model in models:
            file_path = os.path.join(base_dir, f"{model}_{attr}.json")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # ----- neutral distribution -----
            neutral_counts = np.array(
                [data["neutral"].get(k, 0) for k in keys],
                dtype=float
            )
            neutral_sum = neutral_counts.sum()

            if neutral_sum == 0:
                print(f"[Warning] neutral sum is 0: {file_path}")
                continue

            p_neutral = neutral_counts / neutral_sum

            # ----- each emotion vs neutral -----
            for emotion in emotions:
                emo_counts = np.array(
                    [data[emotion].get(k, 0) for k in keys],
                    dtype=float
                )
                emo_sum = emo_counts.sum()

                if emo_sum == 0:
                    print(f"[Warning] emotion sum is 0: {model} / {attr} / {emotion}")
                    continue

                p_emo = emo_counts / emo_sum
                tvd = compute_tvd(p_emo, p_neutral)

                emotion_tvds[emotion].append(tvd)

    # =========================
    # 4. Average TVD per emotion
    # =========================
    mean_tvd = {}
    for emotion in emotions:
        vals = emotion_tvds[emotion]
        if len(vals) == 0:
            mean_tvd[emotion] = np.nan
        else:
            mean_tvd[emotion] = float(np.mean(vals))

    # =========================
    # 5. Rank emotions by mean TVD
    # =========================
    ranked = sorted(
        [(emo, val) for emo, val in mean_tvd.items() if np.isfinite(val)],
        key=lambda x: x[1]
    )

    # =========================
    # 6. Print results
    # =========================
    print("=== Claim (c): Happiness produces the smallest demographic shift ===\n")

    print("Mean TVD by emotion (averaged across all models and attributes):")
    for emo, val in ranked:
        print(f"{emo:>10s}: {val:.4f}")

    print("\nRank order (smallest to largest shift):")
    print(" < ".join([emo for emo, _ in ranked]))

    if len(ranked) > 0 and ranked[0][0] == "happy":
        print("\nConclusion: Happiness has the smallest average demographic shift.")
    else:
        print("\nConclusion: Happiness is NOT the smallest on average.")

    return {
        "emotion_tvds": emotion_tvds,
        "mean_tvd": mean_tvd,
        "ranked": ranked
    }


if __name__ == "__main__":
    results = test_claim_c_happiness_smallest_shift()