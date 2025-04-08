import os
import matplotlib.pyplot as plt
import numpy as np

# 定义多个 active 文件夹
base_dirs = [
    r"./RBC_Dataset/active_learning02/result",
    r"./RBC_Dataset/active_learning03/result",
    r"./RBC_Dataset/active_learning04/result",
    r"./RBC_Dataset/active_learning05/result",
    r"./RBC_Dataset/active_learning07/result",
    r"./RBC_Dataset/active_learning08/result",
    r"./RBC_Dataset/active_learning09/result",
]

base_dirs2 = [
    r"./RBC_Dataset/active_learning02/randomchoose/result",
    r"./RBC_Dataset/active_learning03/randomchoose/result",
    r"./RBC_Dataset/active_learning04/randomchoose/result",
    r"./RBC_Dataset/active_learning05/randomchoose/result",
    r"./RBC_Dataset/active_learning07/randomchoose/result",
    r"./RBC_Dataset/active_learning08/randomchoose/result",
    r"./RBC_Dataset/active_learning09/randomchoose/result",
]

# 定义所有子文件夹
folders = [str(i) for i in range(1, 12)]
folders2 = [str(i) for i in range(2, 12)]

# 用于存储所有 active 目录下的 AP50 值
ap50_values = {folder: [] for folder in folders}
ap50_values2 = {folder: [] for folder in folders2}

# 遍历所有 active 目录
for base_dir in base_dirs:
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        file_path = os.path.join(folder_path, "evaluation_results.txt")

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                lines = f.readlines()
                if len(lines) >= 4:
                    ap50_line = lines[3]
                    if "AP50" in ap50_line:
                        ap50 = float(ap50_line.split("AP50:")[1].split(",")[0].strip())
                        ap50_values[folder].append(ap50)

for base_dir in base_dirs2:
    for folder in folders2:
        folder_path = os.path.join(base_dir, folder)
        file_path = os.path.join(folder_path, "evaluation_results.txt")

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                lines = f.readlines()
                if len(lines) >= 4:
                    ap50_line = lines[3]
                    if "AP50" in ap50_line:
                        ap50 = float(ap50_line.split("AP50:")[1].split(",")[0].strip())
                        ap50_values2[folder].append(ap50)

# 计算每个子目录的 AP50 均值和标准差
average_ap50 = {"normal": [], "com": []}
std_ap50 = {"normal": [], "com": []}

for folder in folders:
    if ap50_values[folder]:
        avg_ap50 = np.mean(ap50_values[folder])
        std_dev_ap50 = np.std(ap50_values[folder]) / 2
        average_ap50["normal"].append(avg_ap50)
        std_ap50["normal"].append(std_dev_ap50)

for folder in folders2:
    if ap50_values2[folder]:
        avg_ap50 = np.mean(ap50_values2[folder])
        std_dev_ap50 = np.std(ap50_values2[folder]) / 2
        if folder == "2":
            average_ap50["com"].append(average_ap50["normal"][0])
            std_ap50["com"].append(std_ap50["normal"][0])
        average_ap50["com"].append(avg_ap50)
        std_ap50["com"].append(std_dev_ap50)

# 生成 X 轴
x_values = [20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80]

# 创建透明背景的图像
plt.figure(figsize=(12, 8), dpi=800, facecolor='none')
plt.gca().set_facecolor('none')

# 绘制曲线
plt.plot(x_values, average_ap50["normal"], marker='o')
plt.fill_between(x_values,
                 np.array(average_ap50["normal"]) - np.array(std_ap50["normal"]),
                 np.array(average_ap50["normal"]) + np.array(std_ap50["normal"]),
                 color='b', alpha=0.2)

plt.plot(x_values, average_ap50["com"], marker='s')
plt.fill_between(x_values,
                 np.array(average_ap50["com"]) - np.array(std_ap50["com"]),
                 np.array(average_ap50["com"]) + np.array(std_ap50["com"]),
                 color='r', alpha=0.2)

# 隐藏坐标轴、边框、标题、标签等
plt.axis('off')

# 保存 PNG，确保背景透明
plt.savefig("RBC_segmentation_ap50_transparent.png", dpi=800, bbox_inches='tight', transparent=True)
plt.close()
