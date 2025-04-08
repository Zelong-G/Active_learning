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
                    # 提取第四行的 AP50 值
                    ap50_line = lines[3]
                    if "AP50" in ap50_line:
                        ap50 = float(ap50_line.split("AP50:")[1].split(",")[0].strip())
                        ap50_values[folder].append(ap50)

# 遍历所有 active 目录
for base_dir in base_dirs2:
    for folder in folders2:
        folder_path = os.path.join(base_dir, folder)
        file_path = os.path.join(folder_path, "evaluation_results.txt")

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                lines = f.readlines()
                if len(lines) >= 4:
                    # 提取第四行的 AP50 值
                    ap50_line = lines[3]
                    if "AP50" in ap50_line:
                        ap50 = float(ap50_line.split("AP50:")[1].split(",")[0].strip())
                        ap50_values2[folder].append(ap50)

# 计算每个子目录的 AP50 均值和标准差
average_ap50 = {"normal": [], "com": []}
std_ap50 = {"normal": [], "com": []}

for folder in folders:
    if ap50_values[folder]:  # 确保有数据
        avg_ap50 = np.mean(ap50_values[folder])  # 计算均值
        std_dev_ap50 = np.std(ap50_values[folder])/2  # 计算标准差
        average_ap50["normal"].append(avg_ap50)
        std_ap50["normal"].append(std_dev_ap50)

for folder in folders2:
    if ap50_values2[folder]:  # 确保有数据
        avg_ap50 = np.mean(ap50_values2[folder])  # 计算均值
        std_dev_ap50 = np.std(ap50_values2[folder])/2  # 计算标准差
        if folder == "2":
            average_ap50["com"].append(average_ap50["normal"][0])
            std_ap50["com"].append(std_ap50["normal"][0])
        average_ap50["com"].append(avg_ap50)
        std_ap50["com"].append(std_dev_ap50)

# 生成 X 轴
x_values = [20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80]  # 过滤 X 轴数据

# 绘制均值 ± 标准差图表
plt.figure(figsize=(12, 8))
plt.plot(x_values, average_ap50["normal"], label="Active Learning", marker='o')
plt.fill_between(x_values,
                 np.array(average_ap50["normal"]) - np.array(std_ap50["normal"]),
                 np.array(average_ap50["normal"]) + np.array(std_ap50["normal"]),
                 color='b', alpha=0.2)

plt.plot(x_values, average_ap50["com"], label="Randomly Choosing", marker='s')
plt.fill_between(x_values,
                 np.array(average_ap50["com"]) - np.array(std_ap50["com"]),
                 np.array(average_ap50["com"]) + np.array(std_ap50["com"]),
                 color='r', alpha=0.2)

# 图表细节设置
plt.xlabel("Size of Training Set",size=18,fontweight='bold')
plt.ylabel("Segmentation AP50 (Mean ± Std/2)",size=18,fontweight='bold')
plt.title("RBC Segmentation AP50 Comparison (Mean ± Std/2 Dev Over 8 Experiments) ",size=18,fontweight='bold')
plt.legend()
plt.grid(True)
# 保存图片，格式可以是 'png', 'pdf', 'svg' 等
plt.savefig("RBC_segmentation_ap50.png", dpi=800, bbox_inches='tight')

plt.show()

