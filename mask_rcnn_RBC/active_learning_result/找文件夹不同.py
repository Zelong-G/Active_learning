
import os
import shutil
# 设置两个文件夹路径

output_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\000"
import os

# 定义文件夹路径
folder1 = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\train2017_00"
folder2 = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\train2017_01"
xx_folder = "/path/to/xx_folder"

# 获取两个文件夹的 .jpg 文件集合
files1 = {f for f in os.listdir(folder1) if f.endswith('.jpg')}
files2 = {f for f in os.listdir(folder2) if f.endswith('.jpg')}

# 计算不共有的文件
unique_files = (files1 - files2) | (files2 - files1)

# 在 xx_folder 中删除不共有的文件
for file in unique_files:
    file_path = os.path.join(xx_folder, file)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file}")
    else:
        print(f"File not found in xx_folder: {file}")

print("操作完成！不共有的文件已从 xx_folder 中删除。")
