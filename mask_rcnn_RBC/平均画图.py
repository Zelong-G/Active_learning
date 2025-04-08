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
missing_folders = []

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
        else:
            missing_folders.append(f"{base_dir}/{folder}")

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
        else:
            missing_folders.append(f"{base_dir}/{folder}")


# 计算每个子目录的 AP50 均值
average_ap50 = {"normal": [], "com": []}

for folder in folders:
    if ap50_values[folder]:  # 确保有数据
        print(ap50_values[folder])
        avg_ap50 = np.mean(ap50_values[folder])  # 计算均值
        print(avg_ap50)
        if folder == "1":  # `1` 文件夹的结果用于两条曲线
            average_ap50["normal"].append(avg_ap50)
        else:
            average_ap50["normal"].append(avg_ap50)
for folder in folders2:
    if ap50_values2[folder]:  # 确保有数据
        print('random:', ap50_values2[folder])
        avg_ap50 = np.mean(ap50_values2[folder])  # 计算均值
        print('random:', avg_ap50)
        if folder == "2":
            average_ap50["com"].append(average_ap50["normal"][0])
        average_ap50["com"].append(avg_ap50)
# 生成 X 轴，仅保留偶数编号：100, 120, 140, 160, 180, 200
full_x_values = list(range(100, 210, 10))  # 原始 X 轴：100, 110, ..., 200


filtered_normal = average_ap50["normal"]
print(filtered_normal)
filtered_com = average_ap50["com"]
print(filtered_com)
x_values = [20,26,32,38,44,50,56,62,68,74,80]  # 过滤 X 轴数据
# # Plot
# plt.figure(figsize=(12, 8))
# 绘制图表
plt.plot(x_values, filtered_normal, label="active learning", marker='o')
plt.plot(x_values, filtered_com, label="randomly choosing", marker='s')
# Styling
plt.gca().set_facecolor('#f5f5f5')
plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='gray')


# 图表细节设置
plt.xlabel("Size of training set")
plt.ylabel("Segmentation AP50 (Averaged)")
plt.title("Segmentation AP50 Comparison (Averaged Over 8 times experiments)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
