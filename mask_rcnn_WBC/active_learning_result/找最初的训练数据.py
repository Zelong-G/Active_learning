import os
import shutil

# 配置路径
folder1 = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell\train2017label"  # 文件夹1路径
folder2 = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell\improvelabel"  # 文件夹2路径
output_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell\train2018label"  # 不重复文件存储的目标文件夹路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 获取两个文件夹中的文件名集合
files_in_folder1 = set(os.listdir(folder1))
files_in_folder2 = set(os.listdir(folder2))

# 找到文件夹1中不在文件夹2中的文件
unique_files = files_in_folder1 - files_in_folder2

# 复制这些文件到目标文件夹
for file in unique_files:
    source_path = os.path.join(folder1, file)
    target_path = os.path.join(output_folder, file)
    shutil.copy(source_path, target_path)

print(f"复制完成！共复制了 {len(unique_files)} 个不重复的文件到 {output_folder}。")
