import os
import matplotlib.pyplot as plt
import numpy as np

# 定义多个 active 文件夹
base_dirs = [
    r"D:\work\Active_learning\mask_rcnn\run_result\active1",
    r"D:\work\Active_learning\mask_rcnn\run_result\active2",
    r"D:\work\Active_learning\mask_rcnn\run_result\active3",
    # r"D:\work\Active_learning\mask_rcnn\run_result\active5",
    # r"D:\work\Active_learning\mask_rcnn\run_result\active6",

]

# 定义所有子文件夹
folders = [str(i) for i in range(1, 12,2)] + [f"{i}com" for i in range(1, 12,2)]

# 用于存储所有 active 目录下的 AP50 值
ap50_values = {folder: [] for folder in folders}
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
print(ap50_values)


# 计算每个子目录的 AP50 均值
average_ap50 = {"normal": [], "com": []}

for folder in folders:
    if ap50_values[folder]:  # 确保有数据
        avg_ap50 = np.mean(ap50_values[folder])  # 计算均值
        if folder == "1":  # `1` 文件夹的结果用于两条曲线
            average_ap50["normal"].append(avg_ap50)
            average_ap50["com"].append(avg_ap50)
        elif "com" in folder:
            average_ap50["com"].append(avg_ap50)
        else:
            average_ap50["normal"].append(avg_ap50)
print(average_ap50)



base_dirs = [

    r"D:\work\Active_learning\mask_rcnn\run_result\active5",
    r"D:\work\Active_learning\mask_rcnn\run_result\active7",

]
folders1 = [str(i) for i in range(1, 7)] + [f"{i}random_com" for i in range(1, 7)]
# 用于存储所有 active 目录下的 AP50 值
ap50_values1 = {folder: [] for folder in folders1}
# 遍历所有 active 目录
for base_dir in base_dirs:
    for folder in folders1:
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
                        ap50_values1[folder].append(ap50)
        else:
            missing_folders.append(f"{base_dir}/{folder}")
print(ap50_values1)


# 计算每个子目录的 AP50 均值
average_ap501 = {"normal": [], "com": []}

for folder in folders1:
    if ap50_values1[folder]:  # 确保有数据
        avg_ap50 = np.mean(ap50_values1[folder])  # 计算均值
        if folder == "1":  # `1` 文件夹的结果用于两条曲线
            average_ap501["normal"].append(avg_ap50)
            average_ap501["com"].append(avg_ap50)
        elif "com" in folder:
            average_ap501["com"].append(avg_ap50)
        else:
            average_ap501["normal"].append(avg_ap50)
print(average_ap501)


import operator
average_ap50["normal"]=list(map(lambda a, b:(a+b)/2, average_ap50["normal"],average_ap501["normal"]))
average_ap50["com"]=list(map(lambda a, b:(a+b)/2, average_ap50["com"],average_ap501["com"]))
print(average_ap50)

# 生成 X 轴，仅保留偶数编号：100, 120, 140, 160, 180, 200
full_x_values = list(range(100, 220, 10))  # 原始 X 轴：100, 110, ..., 200
selected_indices = [i for i, x in enumerate(full_x_values) if x % 20 != 0]  # 仅保留偶数编号的索引

x_values = [full_x_values[i] for i in selected_indices]  # 过滤 X 轴数据
x_values = [100,120,140,160,180,200]  # 过滤 X 轴数据
# filtered_normal = [average_ap50["normal"][i] for i in selected_indices]
# filtered_com = [average_ap50["com"][i] for i in selected_indices]
filtered_normal = average_ap50["normal"]
filtered_com = average_ap50["com"]
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
plt.title("Segmentation AP50 Comparison (Averaged Over 3 times experiments)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
