import os
import matplotlib.pyplot as plt
import numpy as np

# 定义多个 active 文件夹
base_dirs = [
    r"D:\work\Active_learning\mask_rcnn\run_result\active1",
    r"D:\work\Active_learning\mask_rcnn\run_result\active2",
    # r"D:\work\Active_learning\mask_rcnn\run_result\active3",
]

# 定义所有子文件夹
folders = [str(i) for i in range(1, 12, 2)] + [f"{i}com" for i in range(1, 12, 2)]

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

# 计算每个子目录的 AP50 均值和标准差
average_ap50 = {"normal": [], "com": []}
std_ap50 = {"normal": [], "com": []}

for folder in folders:
    if ap50_values[folder]:  # 确保有数据
        avg_ap50 = np.mean(ap50_values[folder])  # 计算均值
        std_dev = np.std(ap50_values[folder])  # 计算标准差
        if folder == "1":
            average_ap50["normal"].append(avg_ap50)
            std_ap50["normal"].append(std_dev)
            average_ap50["com"].append(avg_ap50)
            std_ap50["com"].append(std_dev)
        elif "com" in folder:
            average_ap50["com"].append(avg_ap50)
            std_ap50["com"].append(std_dev)
        else:
            average_ap50["normal"].append(avg_ap50)
            std_ap50["normal"].append(std_dev)

base_dirs = [
    r"D:\work\Active_learning\mask_rcnn\run_result\active5",
    r"D:\work\Active_learning\mask_rcnn\run_result\active7",
]

folders1 = [str(i) for i in range(1, 7)] + [f"{i}random_com" for i in range(1, 7)]
ap50_values1 = {folder: [] for folder in folders1}

for base_dir in base_dirs:
    for folder in folders1:
        folder_path = os.path.join(base_dir, folder)
        file_path = os.path.join(folder_path, "evaluation_results.txt")

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                lines = f.readlines()
                if len(lines) >= 4:
                    ap50_line = lines[3]
                    if "AP50" in ap50_line:
                        ap50 = float(ap50_line.split("AP50:")[1].split(",")[0].strip())
                        ap50_values1[folder].append(ap50)
        else:
            missing_folders.append(f"{base_dir}/{folder}")

# 计算 AP50 均值和标准差
average_ap501 = {"normal": [], "com": []}
std_ap501 = {"normal": [], "com": []}

for folder in folders1:
    if ap50_values1[folder]:
        avg_ap50 = np.mean(ap50_values1[folder])
        std_dev = np.std(ap50_values1[folder])
        if folder == "1":
            average_ap501["normal"].append(avg_ap50)
            std_ap501["normal"].append(std_dev)
            average_ap501["com"].append(avg_ap50)
            std_ap501["com"].append(std_dev)
        elif "com" in folder:
            average_ap501["com"].append(avg_ap50)
            std_ap501["com"].append(std_dev)
        else:
            average_ap501["normal"].append(avg_ap50)
            std_ap501["normal"].append(std_dev)

# 合并数据 (均值和标准差)
average_ap50["normal"] = list(map(lambda a, b: (a + b) / 2, average_ap50["normal"], average_ap501["normal"]))
average_ap50["com"] = list(map(lambda a, b: (a + b) / 2, average_ap50["com"], average_ap501["com"]))

std_ap50["normal"] = list(map(lambda a, b: np.sqrt((a**2 + b**2) / 2), std_ap50["normal"], std_ap501["normal"]))
std_ap50["com"] = list(map(lambda a, b: np.sqrt((a**2 + b**2) / 2), std_ap50["com"], std_ap501["com"]))

# X 轴数据
x_values = [100, 120, 140, 160, 180, 200]

# 绘制均值和标准差的误差棒图
plt.figure(figsize=(12, 8))
plt.plot(x_values, average_ap50["normal"], label="Active Learning", marker='o')
plt.fill_between(x_values,
                 np.array(average_ap50["normal"]) - np.array(std_ap50["normal"])/2,
                 np.array(average_ap50["normal"]) + np.array(std_ap50["normal"])/2,
                 color='b', alpha=0.2)
plt.plot(x_values, average_ap50["com"], label="Randomly Choosing", marker='s')
plt.fill_between(x_values,
                 np.array(average_ap50["com"]) - np.array(std_ap50["com"])/2,
                 np.array(average_ap50["com"]) + np.array(std_ap50["com"])/2,
                 color='r', alpha=0.2)


# 细节设置
# 图表细节设置
# 隐藏坐标轴、边框、标题、标签等
plt.axis('off')

# 保存 PNG，确保背景透明
plt.savefig("WBC_segmentation_ap50_transparent.png", dpi=800, bbox_inches='tight', transparent=True)
plt.close()
