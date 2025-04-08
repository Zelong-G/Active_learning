import os
import matplotlib.pyplot as plt

# 定义文件夹路径和范围
base_dir = r"./RBC_Dataset/active_learning07/result"  # 替换为实际的父文件夹路径
folders = [str(i) for i in range(1, 12)]

# 提取AP50值
results = {"normal": []}
missing_folders = []

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
                    if folder == "1":  # `1` 文件夹的结果用在两条线上
                        results["normal"].append(ap50)

                    else:
                        results["normal"].append(ap50)
            else:
                missing_folders.append(folder)  # 第四行不存在
    else:
        missing_folders.append(folder)  # 文件不存在
# 绘制图表
x_values = list(range(100, 210, 10))  # X轴从100到200，步长10
plt.plot(x_values, results["normal"], label="Normal")

# 图表细节设置
plt.xlabel("X-axis (100 to 200)")
plt.ylabel("Segmentation AP50")
plt.title("Segmentation AP50 Comparison")
plt.legend()
plt.grid(True)
print(results)
plt.show()
