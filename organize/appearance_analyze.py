import os
import pandas as pd


def compare_attractive_plain():

    root_dir = "/Volumes/KINGSTON/projects/bias"

    models = [
        "flux",
        "hunyuan",
        "kolors",
        "proteus",
        "qwen",
        "sana",
        "sd3",
        "wan2.1"
    ]

    def compute_stats(csv_path):

        df = pd.read_csv(csv_path)

        total = len(df)

        counts = df["attractiveness"].value_counts()

        c0 = counts.get(0, 0)
        c1 = counts.get(1, 0)
        c2 = counts.get(2, 0)

        ratio_0 = c0 / total
        ratio_1 = c1 / total
        ratio_2 = c2 / total

        # 计算总分
        score = c0 * 0 + c1 * 1 + c2 * 2

        return ratio_0, ratio_1, ratio_2, score


    results = []

    for model in models:

        appearance_dir = os.path.join(root_dir, model, "appearance")

        attractive_csv = os.path.join(appearance_dir, "attractive_res.csv")
        plain_csv = os.path.join(appearance_dir, "plain_res.csv")

        a0, a1, a2, attr_score = compute_stats(attractive_csv)
        p0, p1, p2, plain_score = compute_stats(plain_csv)

        results.append({
            "model": model,

            "attr_0": a0,
            "attr_1": a1,
            "attr_2": a2,
            "attr_score": attr_score,

            "plain_0": p0,
            "plain_1": p1,
            "plain_2": p2,
            "plain_score": plain_score
        })

    result_df = pd.DataFrame(results)

    print(result_df)

    result_df.to_csv("attractiveness_comparison.csv", index=False)

if __name__ == '__main__':
    compare_attractive_plain()
    print("done")
