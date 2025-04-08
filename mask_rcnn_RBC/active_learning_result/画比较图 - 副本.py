import os
import matplotlib.pyplot as plt
import numpy as np

# 定义多个active文件夹
base_dirs = [
    r"D:\work\Active_learning\mask_rcnn\run_result\active1",
    r"D:\work\Active_learning\mask_rcnn\run_result\active2",
    r"D:\work\Active_learning\mask_rcnn\run_result\active3"
]

# 定义所有子文件夹
folders = [str(i) for i in range(1, 12)] + [f"{i}com" for i in range(1, 12)]

# 用于存储所有active目录下的AP50值
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

# 计算每个子目录的AP50均值
average_ap50 = {"normal": [], "com": []}

for folder in folders:
    if ap50_values[folder]:  # 确保不为空
        avg_ap50 = np.mean(ap50_values[folder])  # 计算平均值
        if folder == "1":  # `1` 文件夹的结果用于两条曲线
            average_ap50["normal"].append(avg_ap50)
            average_ap50["com"].append(avg_ap50)
        elif "com" in folder:
            average_ap50["com"].append(avg_ap50)
        else:
            average_ap50["normal"].append(avg_ap50)

# 生成X轴
x_values = list(range(100, 210, 10))  # X轴从100到200，步长10

# 绘制图表
plt.plot(x_values, average_ap50["normal"], label="Normal", marker='o')
plt.plot(x_values, average_ap50["com"], label="Com", marker='s')

# 图表细节设置
plt.xlabel("X-axis (100 to 200)")
plt.ylabel("Segmentation AP50 (Averaged)")
plt.title("Segmentation AP50 Comparison (Averaged Over Active1-3)")
plt.legend()
plt.grid(True)
plt.show()
