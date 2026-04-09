import os
import json
import numpy as np
from scipy.stats import chi2_contingency

def table_age_emotion_older_representation():
    # =========================
    # 1. Config
    # =========================
    base_dir = "/Users/wmt/projects/bias/write_into_json/age"
    models = [
        "flux", "proteus", "sd3", "sana",
        "hunyuan", "qwen", "kolors", "wan2.1"
    ]
    age_keys = ["0-9", "10-19", "20-39", "40-59", "60+"]
    emotions = ["angry", "disgusted"]

    # =========================
    # 2. Pool counts across all models
    # =========================
    pooled = {
        "neutral":   np.zeros(len(age_keys), dtype=int),
        "angry":     np.zeros(len(age_keys), dtype=int),
        "disgusted": np.zeros(len(age_keys), dtype=int),
    }

    for model in models:
        file_path = os.path.join(base_dir, f"{model}_age.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for emotion in pooled:
            counts = np.array([data[emotion].get(k, 0) for k in age_keys], dtype=int)
            pooled[emotion] += counts

    # =========================
    # 3. Print table + chi-square for each emotion
    # =========================
    for emotion in emotions:
        pooled_emotion = pooled[emotion]
        pooled_neutral = pooled["neutral"]

        emotion_total = pooled_emotion.sum()
        neutral_total = pooled_neutral.sum()

        # Chi-squared test
        contingency_table = np.array([pooled_emotion, pooled_neutral])
        chi2, p, dof, _ = chi2_contingency(contingency_table)

        # Header
        print(f"\n=== Emotion: {emotion.capitalize()} vs Neutral ===\n")
        col_w = 18
        header = f"{'Condition':<12}" + "".join(f"{k:>{col_w}}" for k in age_keys)
        print(header)
        print("-" * (12 + col_w * len(age_keys)))

        # Neutral row
        neutral_row = f"{'Neutral':<12}"
        for i, k in enumerate(age_keys):
            cell = f"{pooled_neutral[i]:,} ({pooled_neutral[i]/neutral_total*100:.2f}%)"
            neutral_row += f"{cell:>{col_w}}"
        print(neutral_row)

        # Emotion row
        emotion_row = f"{emotion.capitalize():<12}"
        for i, k in enumerate(age_keys):
            cell = f"{pooled_emotion[i]:,} ({pooled_emotion[i]/emotion_total*100:.2f}%)"
            emotion_row += f"{cell:>{col_w}}"
        print(emotion_row)

        print("-" * (12 + col_w * len(age_keys)))

        # Delta row
        delta_row = f"{'Δ':<12}"
        for i in range(len(age_keys)):
            delta = pooled_emotion[i]/emotion_total - pooled_neutral[i]/neutral_total
            sign = "+" if delta >= 0 else ""
            cell = f"{sign}{delta*100:.2f}pp"
            delta_row += f"{cell:>{col_w}}"
        print(delta_row)

        # Chi-square result
        print(f"\nChi-square: {chi2:.4f}, df={dof}, p={p:.4e}")
        sig = "statistically significant (p < 0.05)" if p < 0.05 else "NOT significant (p >= 0.05)"
        print(f"Distribution shift is {sig}.")


if __name__ == "__main__":
    table_age_emotion_older_representation()
