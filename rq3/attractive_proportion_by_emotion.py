import numpy as np
import pandas as pd
import os

# =========================
# 1. 配置
# =========================
base_path = "/Users/wmt/projects/bias/data/corrected"

models = ["flux", "hunyuan", "kolors", "proteus", "qwen", "sana", "sd3", "wan2.1"]

emotions = ["neutral", "angry", "disgusted", "fearful", "happy", "sad", "surprised"]

# attractiveness: 0=low, 1=medium, 2=high
attr_labels = {0: "low", 1: "medium", 2: "high"}

# =========================
# 2. 读取所有数据
# =========================
def load_attractiveness(model, emotion):
    path = os.path.join(base_path, model, f"{emotion}_res.csv")
    df = pd.read_csv(path)
    # 只取face0
    df = df[df["face_name_align"].str.contains("face0", na=False)]
    return df["attractiveness"].values

# =========================
# 3. 计算每个 emotion × attractiveness 的比例
# =========================
# result[emotion][level] = 8个模型的均值比例
result = {}

for emotion in emotions:
    level_props = {0: [], 1: [], 2: []}  # 每个level收集8个模型的比例

    for model in models:
        try:
            attrs = load_attractiveness(model, emotion)
            total = len(attrs)
            for level in [0, 1, 2]:
                prop = np.sum(attrs == level) / total
                level_props[level].append(prop)
        except FileNotFoundError:
            print(f"WARNING: {model}/{emotion}_res.csv not found, skipping")

    result[emotion] = {
        level: np.mean(props) for level, props in level_props.items()
    }

# =========================
# 4. 打印表格
# =========================
print("\n3×7 Attractiveness Table (mean proportion across 8 models):\n")
header = f"{'':10}" + "".join(f"{e:>12}" for e in emotions)
print(header)
print("-" * (10 + 12 * len(emotions)))

for level in [0, 1, 2]:
    label = attr_labels[level]
    row = f"{label:10}" + "".join(
        f"{result[e][level]*100:>11.1f}%" for e in emotions
    )
    print(row)

# =========================
# 5. 输出报告语句
# =========================
neutral_low    = result["neutral"][0]
neutral_high   = result["neutral"][2]
anger_low      = result["angry"][0]
happy_high     = result["happy"][2]

print("\n" + "=" * 65)
print("RESULT:")
print(
    f'Under Anger, {anger_low*100:.1f}% of generated faces were rated '
    f'low-attractiveness (neutral baseline: {neutral_low*100:.1f}%); '
    f'under Happiness, {happy_high*100:.1f}% were rated '
    f'high-attractiveness (neutral baseline: {neutral_high*100:.1f}%).'
)
print("=" * 65)

import numpy as np

# neutral基准分布
neutral_dist = np.array([result["neutral"][0],
                         result["neutral"][1],
                         result["neutral"][2]])

print("TVD (attractiveness vs neutral baseline):\n")
for emotion in ["angry", "disgusted", "fearful", "happy", "sad", "surprised"]:
    emotion_dist = np.array([result[emotion][0],
                             result[emotion][1],
                             result[emotion][2]])
    tvd = 0.5 * np.sum(np.abs(emotion_dist - neutral_dist))
    print(f"  {emotion:12}: TVD = {tvd:.4f}")



if __name__ == '__main__':
    print("done")