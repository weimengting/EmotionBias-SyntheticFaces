import pandas as pd
import os

# =========================
# 1. Load CSV
# =========================
df = pd.read_csv("/Users/wmt/projects/bias/data/res/kolors/res.csv")

# =========================
# 2. Keep only face0
# =========================
df = df[df["face_name_align"].str.contains("face0", na=False)].copy()

# =========================
# 3. Parse ground-truth attributes from path
# =========================
def parse_gt_from_path(path):
    """
    Example path:
    /scratch/.../selected_faces/0-9/female/asian/0000_face0.png
    """
    parts = path.split(os.sep)

    try:
        # assume structure: ... / age / gender / race / filename
        age = parts[-4]
        gender = parts[-3]
        race = parts[-2]
    except IndexError:
        age, gender, race = None, None, None

    return pd.Series([age, gender, race])

df[["gt_age", "gt_gender", "gt_race"]] = df["face_name_align"].apply(parse_gt_from_path)

# =========================
# 4. (Optional) sanity check
# =========================
print(df[["face_name_align", "gt_age", "gt_gender", "gt_race"]].head())

# =========================
# 5. (Optional) save
# =========================
df.to_csv("/Users/wmt/projects/bias/data/res/kolors/res.csv", index=False)


if __name__ == '__main__':
    print('done')