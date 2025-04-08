import os
import numpy as np
import matplotlib.pyplot as plt

# 读取 classification_results.txt 并提取 macro avg 的 f1-score
def extract_macro_avg_f1(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("   macro avg"):
                    values = line.split()
                    if len(values) >= 3:
                        return float(values[4])  # 第三个值是 f1-score
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

# 基础路径
base_path = "./run_result"
experiment_folders = [f"active{i}" for i in range(5, 8)]  # 02 到 09

# 存储 y 轴数据（11个文件夹的平均F1-score）
folder_f1_scores = {i: [] for i in range(2, 7)}  # 存储每个 folder 1-11 的 f1-score

# 遍历所有实验文件夹
for exp in experiment_folders:
    result_path = os.path.join(base_path, exp)
    for i in range(2, 7):
        file_path = os.path.join(result_path, str(i)+'random_com', "classification_results.txt")
        print(file_path)
        if os.path.exists(file_path):
            f1_score = extract_macro_avg_f1(file_path)
            if f1_score is not None:
                folder_f1_scores[i].append(f1_score)
print(folder_f1_scores)

# 计算平均值
x_values = list(range(2, 7))
y_values = [np.mean(folder_f1_scores[i]) if folder_f1_scores[i] else 0 for i in x_values]


# 基础路径
base_path = "./run_result"
experiment_folders = [f"active{i}" for i in range(5, 8)]  # 02 到 09

# 存储 y 轴数据（11个文件夹的平均F1-score）
folder_f1_scores = {i: [] for i in range(1, 7)}  # 存储每个 folder 1-11 的 f1-score

# 遍历所有实验文件夹
for exp in experiment_folders:
    result_path = os.path.join(base_path, exp)
    for i in range(1, 7):
        file_path = os.path.join(result_path, str(i), "classification_results.txt")
        if os.path.exists(file_path):
            f1_score = extract_macro_avg_f1(file_path)
            if f1_score is not None:
                folder_f1_scores[i].append(f1_score)
print(folder_f1_scores)
# 计算平均值
x_values1 = list(range(1, 7))
y_values1 = [np.mean(folder_f1_scores[i]) if folder_f1_scores[i] else 0 for i in x_values1]
y_values.insert(0, y_values1[0])  # 在索引 0 处插入 1

x_values=[120,140,160,180,200]
x_values1=[100,120,140,160,180,200]
# 绘制图表
plt.figure(figsize=(8, 5))
plt.plot(x_values1, y_values1, marker='o', linestyle='-', color='b', label='Active learning Average F1-score')
plt.plot(x_values1, y_values, marker='o', linestyle='-', color='orange', label='Random Average F1-score')

plt.xlabel('The size of training dataset')
plt.ylabel('Avg F1-score')
plt.title('Average F1-score Across 5 Experiments (WBC)')
plt.xticks([100,120,140,160,180,200])
plt.grid(True)
plt.legend()
plt.show()
