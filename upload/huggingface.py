import os
import shutil


def organize_data():
    models_base = "/Volumes/KINGSTON/bias"
    model_names = ["flux", "proteus", "sd3", "sana",
                   "hunyuan", "kolors", "qwen", "wan2.1"]

    emotions = [
        "angry_persons", "disgusted_persons", "fearful_persons",
        "happy_persons", "persons", "sad_persons",
        "surprised_persons", "unhappy_persons",
    ]

    prefix_map = {
        "angry_persons":    "angry_person_",
        "disgusted_persons":"disgusted_person_",
        "fearful_persons":  "fearful_person_",
        "happy_persons":    "happy_person_",
        "persons":          "person_",
        "sad_persons":      "sad_person_",
        "surprised_persons":"surprised_person_",
        "unhappy_persons":  "unhappy_person_",
    }

    for model in model_names:
        base_path = os.path.join(models_base, model)

        for emotion in emotions:
            src_dir = os.path.join(base_path, "complete", emotion)
            dst_dir = os.path.join(base_path, emotion)
            prefix  = prefix_map[emotion]

            if not os.path.exists(src_dir):
                continue

            src_files = [f for f in os.listdir(src_dir) if f.endswith(".png")]

            for fname in src_files:
                # fname 类似 0125.png，取出编号部分
                stem     = os.path.splitext(fname)[0]   # "0125"
                new_name = f"{prefix}{stem}.png"         # "angry_person_0125.png"
                src      = os.path.join(src_dir, fname)
                dst      = os.path.join(dst_dir, new_name)

                if not os.path.exists(dst):
                    print(f"  新增: {fname} → {new_name}")

                shutil.copy2(src, dst)
                print(f"  替换: {fname} → {new_name}")

                # shutil.copy2(src, dst)   # 覆盖原文件
                # print(f"  替换: {fname} → {new_name}")

        print(f"{model} 完成")

def upload():
    from huggingface_hub import HfApi

    api = HfApi()
    # # 先创建仓库
    # api.create_repo(
    #     repo_id="mengtingwei/emotion_bias",
    #     repo_type="dataset",
    #     private=False  # 或 False 公开
    # )
    # 上传单个文件
    # api.upload_file(
    #     path_or_fileobj="/Volumes/KINGSTON/bias/proteus/proteus.tar.gz",
    #     path_in_repo="proteus.tar.gz",
    #     repo_id="mengtingwei/emotion_bias",
    #     repo_type="dataset",
    # )
    # api.upload_file(
    #     path_or_fileobj="/Volumes/KINGSTON/bias/qwen/qwen.tar.gz",
    #     path_in_repo="qwen.tar.gz",
    #     repo_id="mengtingwei/emotion_bias",
    #     repo_type="dataset",
    # )
    # api.upload_file(
    #     path_or_fileobj="/Volumes/KINGSTON/bias/sd3/sd3.tar.gz",
    #     path_in_repo="sd3.tar.gz",
    #     repo_id="mengtingwei/emotion_bias",
    #     repo_type="dataset",
    # )
    api.upload_file(
        path_or_fileobj="/Volumes/KINGSTON/bias/wan2.1/wan2.1.tar.gz",
        path_in_repo="wan2.1.tar.gz",
        repo_id="mengtingwei/emotion_bias",
        repo_type="dataset",
    )
    # # 上传整个文件夹
    # api.upload_folder(
    #     folder_path="./your_data_folder",
    #     repo_id="your_username/your_repo_name",
    #     repo_type="dataset",
    # )

if __name__ == '__main__':
    upload()
    print("done")