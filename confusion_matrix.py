# compute confusion matrix

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix




def race_cm_5_classes():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # 读取数据
    df = pd.read_csv("/Users/wmt/projects/bias/data/fairface/cfd_omit_middle_eastern_race_output.csv")

    # 如果 race == "Middle Eastern"，用 race4 中的结果替换 adjusted_race
    df.loc[df["race"] == "Middle Eastern", "adjusted_race"] = \
        df.loc[df["race"] == "Middle Eastern", "race4"]

    # 保存结果（如果你确实需要）
    df.to_csv("/Users/wmt/projects/bias/data/fairface/cfd_race_adjusted_fixed.csv", index=False)

    # ---- Step 1: 解析 ground truth (race_from_name) ----
    def map_gt(x):
        x = str(x)
        if "Latino" in x:
            return "Latino"
        if "Indian" in x:
            return "India"
        if "Asian" in x:
            return "Asian"
        return x  # White or Black

    df['gt'] = df['race_from_name'].apply(map_gt)

    # ---- Step 2: 解析预测值 adjusted_race ----
    def map_pred(x):
        x = str(x)
        if "Latino" in x:
            return "Latino"
        if "India" in x:
            return "India"
        if "Asian" in x:
            return "Asian"
        return x

    df['pred'] = df['adjusted_race'].apply(map_pred)

    # ---- Step 3: 生成混淆矩阵 ----
    labels = ["White", "Black", "Latino", "India", "Asian"]
    cm = confusion_matrix(
        df['gt'],
        df['pred'],
        labels=labels,
        normalize='true'
    )

    # ---- Step 3.5: overall accuracy ----
    overall_acc = (df['gt'] == df['pred']).mean()

    # ---- Step 4: 绘图 ----
    plt.figure(figsize=(7, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 16}   # 👈 格子里数字字体
    )

    # 坐标轴标题（变大）
    plt.xlabel("Predicted label", fontsize=16)
    plt.ylabel("True label", fontsize=16)

    # 坐标刻度（变大）
    plt.xticks(fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=90)

    # 总标题（变大）
    # plt.title(
    #     f"CFD Race Confusion Matrix (5 classes)\nAccuracy = {overall_acc:.4f}",
    #     fontsize=16
    # )

    plt.tight_layout()

    # ---- Step 5: 保存图像 ----
    output_path = "./cfd_race_confusion_matrix_5cls.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("Image saved to:", output_path)
    print(f"Overall Accuracy = {overall_acc:.4f}")



def race_cm_4_classes():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # 读取数据（与 5-class 版本路径风格一致）
    df = pd.read_csv(
        "/Users/wmt/projects/bias/data/fairface/cfd_omit_middle_eastern_race_output.csv"
    )

    # ---- Step 1: Ground truth mapping (4 classes) ----
    def map_gt(x):
        x = str(x)
        if "Latino" in x:
            return "Others"
        if "Indian" in x:
            return "Others"
        if "Asian" in x:
            return "Asian"
        return x   # White or Black

    df["gt4"] = df["race_from_name"].apply(map_gt)

    # ---- Step 2: Prediction mapping (4 classes) ----
    def map_pred(x):
        x = str(x)
        if "Latino" in x:
            return "Others"
        if "India" in x:
            return "Others"
        if "Asian" in x:
            return "Asian"
        return x

    df["pred4"] = df["adjusted_race"].apply(map_pred)

    # ---- Step 3: 生成 4-class 混淆矩阵 ----
    labels4 = ["White", "Black", "Others", "Asian"]
    cm4 = confusion_matrix(
        df["gt4"],
        df["pred4"],
        labels=labels4,
        normalize="true"
    )

    # ---- Step 3.5: overall accuracy ----
    overall_acc = (df["gt4"] == df["pred4"]).mean()

    # ---- Step 4: 绘图 ----
    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm4,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels4,
        yticklabels=labels4,
        annot_kws={"size": 16}   # 👈 与 5-class 完全一致
    )

    # 坐标轴标题（对齐 5-class 风格）
    plt.xlabel("Predicted label", fontsize=16)
    plt.ylabel("True label", fontsize=16)

    # 坐标刻度（对齐 5-class 风格）
    plt.xticks(fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=90)

    # 总标题（与 5-class 一致：不显示）
    # plt.title(
    #     f"CFD Race Confusion Matrix (4 classes: Latino+India → Others)\n"
    #     f"Accuracy = {overall_acc:.4f}",
    #     fontsize=16
    # )

    plt.tight_layout()

    # ---- Step 5: 保存图像 ----
    output_path = "./cfd_race_confusion_matrix_4cls.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("Image saved to:", output_path)
    print(f"Overall Accuracy = {overall_acc:.4f}")



def gender_cm():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # 读取数据（与 race_cm 路径风格一致）
    df = pd.read_csv(
        "/Users/wmt/projects/bias/data/fairface/cfd_omit_middle_eastern_race_output.csv"
    )

    # ---- Step 1: Ground truth mapping ----
    def map_gender_gt(x):
        x = str(x).lower()
        if "female" in x:
            return "Female"
        if "male" in x:
            return "Male"
        return x

    df["gt_gender"] = df["gender_from_name"].apply(map_gender_gt)

    # ---- Step 2: Prediction mapping ----
    def map_gender_pred(x):
        x = str(x).lower()
        if "female" in x:
            return "Female"
        if "male" in x:
            return "Male"
        return x

    df["pred_gender"] = df["gender"].apply(map_gender_pred)

    # ---- Step 3: Confusion matrix ----
    gender_labels = ["Female", "Male"]
    cm_gender = confusion_matrix(
        df["gt_gender"],
        df["pred_gender"],
        labels=gender_labels,
        normalize="true"
    )

    # ---- Step 3.5: overall accuracy ----
    overall_acc = (df["gt_gender"] == df["pred_gender"]).mean()

    # ---- Step 4: Plot ----
    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm_gender,
        annot=True,
        fmt=".2f",
        cmap="Purples",          # ✅ 保留原来的颜色
        xticklabels=gender_labels,
        yticklabels=gender_labels,
        annot_kws={"size": 16}
    )

    # 坐标轴标题（与 race_cm 风格一致）
    plt.xlabel("Predicted label", fontsize=16)
    plt.ylabel("True label", fontsize=16)

    # 坐标刻度（与 race_cm 风格一致）
    plt.xticks(fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=90)

    # 总标题（与 race_cm 一致：不显示）
    # plt.title(
    #     f"CFD Gender Confusion Matrix\nAccuracy = {overall_acc:.4f}",
    #     fontsize=16
    # )

    plt.tight_layout()

    # ---- Step 5: Save ----
    output_path_gender = "./cfd_gender_confusion_matrix.png"
    plt.savefig(output_path_gender, dpi=300, bbox_inches='tight')
    plt.show()

    print("Image saved to:", output_path_gender)
    print(f"Overall Accuracy = {overall_acc:.4f}")



def age_cm():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # 读取数据（与其他 CM 路径风格一致）
    df = pd.read_csv(
        "/Users/wmt/projects/bias/data/fairface/fgnet_outputs_with_age_group_merged.csv"
    )

    # ---- Step 1: 将 GT_age 映射为 GT_group（真实年龄段） ----
    def map_age_group(age):
        if age < 10:
            return "0-9"
        elif age < 20:
            return "10-19"
        elif age < 40:
            return "20-39"
        elif age < 60:
            return "40-59"
        else:
            return "60+"

    df["GT_group"] = df["real_age"].apply(map_age_group)

    # ---- Step 2: 预测标签 ----
    df["Pred_group"] = df["age_group_merged"]

    # ---- Step 3: 定义标签顺序 ----
    labels = ["0-9", "10-19", "20-39", "40-59", "60+"]

    # ---- Step 4: Confusion Matrix ----
    cm = confusion_matrix(
        df["GT_group"],
        df["Pred_group"],
        labels=labels,
        normalize="true"
    )

    # ---- Step 5: 计算 overall accuracy ----
    def check_correct(row):
        pred = row["Pred_group"]
        age = row["real_age"]

        if pred == "0-9":
            return age < 10
        elif pred == "10-19":
            return 10 <= age < 20
        elif pred == "20-39":
            return 20 <= age < 40
        elif pred == "40-59":
            return 40 <= age < 60
        else:
            return age >= 60

    accuracy = df.apply(check_correct, axis=1).mean()

    # ---- Step 6: 绘图（统一风格，保留配色）----
    plt.figure(figsize=(7, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",           # ✅ 保留原来的颜色
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 16}
    )

    # 坐标轴标题（统一风格）
    plt.xlabel("Predicted label", fontsize=16)
    plt.ylabel("True label", fontsize=16)

    # 坐标刻度（统一风格）
    plt.xticks(fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=90)

    # 总标题（统一：不显示）
    # plt.title(
    #     f"FGNET Age Confusion Matrix (5 classes)\nAccuracy = {accuracy:.4f}",
    #     fontsize=16
    # )

    plt.tight_layout()

    # ---- Step 7: 保存图片 ----
    output_path = "./fgnet_age_confusion_matrix_5cls.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("Image saved to:", output_path)
    print(f"Accuracy = {accuracy:.4f}")



def age_FACES_cm():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # 读取数据（与其他 CM 路径风格一致）
    df = pd.read_csv(
        "/Users/wmt/projects/bias/data/fairface/FACES_outputs_with_age_group_merged.csv"
    )

    # ---- Step 1: 将 GT_age 映射为 GT_group（真实年龄段） ----
    def map_age_group(age):
        if age < 10:
            return "0-9"
        elif age < 20:
            return "10-19"
        elif age < 40:
            return "20-39"
        elif age < 60:
            return "40-59"
        else:
            return "60+"

    df["GT_group"] = df["GT_age"].apply(map_age_group)

    # ---- Step 2: 预测标签 ----
    df["Pred_group"] = df["age_group_merged"]

    # ---- Step 3: 定义标签顺序 ----
    labels = ["0-9", "10-19", "20-39", "40-59", "60+"]

    # ---- Step 4: Confusion Matrix ----
    cm = confusion_matrix(
        df["GT_group"],
        df["Pred_group"],
        labels=labels,
        normalize="true"
    )

    # ---- Step 5: 计算 overall accuracy ----
    def check_correct(row):
        pred = row["Pred_group"]
        age = row["GT_age"]

        if pred == "0-9":
            return age < 10
        elif pred == "10-19":
            return 10 <= age < 20
        elif pred == "20-39":
            return 20 <= age < 40
        elif pred == "40-59":
            return 40 <= age < 60
        else:
            return age >= 60

    accuracy = df.apply(check_correct, axis=1).mean()

    # ---- Step 6: 绘图（统一风格，保留配色）----
    plt.figure(figsize=(7, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",           # ✅ 保留原来的颜色
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 16}
    )

    # 坐标轴标题（统一风格）
    plt.xlabel("Predicted label", fontsize=16)
    plt.ylabel("True label", fontsize=16)

    # 坐标刻度（统一风格）
    plt.xticks(fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=90)

    # 总标题（统一：不显示）
    # plt.title(
    #     f"FGNET Age Confusion Matrix (5 classes)\nAccuracy = {accuracy:.4f}",
    #     fontsize=16
    # )

    plt.tight_layout()

    # ---- Step 7: 保存图片 ----
    output_path = "./FACES_age_confusion_matrix_5cls.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("Image saved to:", output_path)
    print(f"Accuracy = {accuracy:.4f}")

if __name__ == '__main__':
    race_cm_5_classes()
    race_cm_4_classes()
    gender_cm()
    age_cm()
    age_FACES_cm()