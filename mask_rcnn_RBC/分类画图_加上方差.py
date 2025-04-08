import os
import numpy as np
import matplotlib.pyplot as plt

def extract_macro_avg_f1(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("   macro avg"):
                    values = line.split()
                    if len(values) >= 5:
                        return float(values[4])  # 第五个值是 f1-score
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

# 基础路径
base_path = "./RBC_Dataset"
experiment_folders = [f"active_learning0{i}" for i in range(2, 9)]  # 02 到 09

# 存储 y 轴数据（11个文件夹的 F1-score）
folder_f1_scores = {i: [] for i in range(2, 12)}  # 存储每个 folder 2-11 的 f1-score

# 遍历所有实验文件夹（随机选择）
for exp in experiment_folders:
    result_path = os.path.join(base_path, exp, "randomchoose/result")
    for i in range(2, 12):
        file_path = os.path.join(result_path, str(i), "classification_results.txt")
        if os.path.exists(file_path):
            f1_score = extract_macro_avg_f1(file_path)
            if f1_score is not None:
                folder_f1_scores[i].append(f1_score)

# 计算均值和标准差
x_values = list(range(2, 12))
y_means = [np.mean(folder_f1_scores[i]) if folder_f1_scores[i] else 0 for i in x_values]
y_stds = [np.std(folder_f1_scores[i]) if folder_f1_scores[i] else 0 for i in x_values]

# 重新存储 active learning 结果
folder_f1_scores = {i: [] for i in range(1, 12)}  # 存储每个 folder 1-11 的 f1-score

# 遍历所有实验文件夹（主动学习）
for exp in experiment_folders:
    result_path = os.path.join(base_path, exp, "result")
    for i in range(1, 12):
        file_path = os.path.join(result_path, str(i), "classification_results.txt")
        if os.path.exists(file_path):
            f1_score = extract_macro_avg_f1(file_path)
            if f1_score is not None:
                folder_f1_scores[i].append(f1_score)

# 计算均值和标准差
x_values1 = list(range(1, 12))
y_means1 = [np.mean(folder_f1_scores[i]) if folder_f1_scores[i] else 0 for i in x_values1]
y_stds1 = [np.std(folder_f1_scores[i]) if folder_f1_scores[i] else 0 for i in x_values1]

y_means.insert(0, y_means1[0])  # 在索引 0 处插入 1
y_stds.insert(0, y_stds1[0])
print(y_means, y_stds)
# 显示的数据集大小
x_values = [26, 32, 38, 44, 50, 56, 62, 68, 74, 80]
x_values1 = [20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80]

# 绘制图表
plt.figure(figsize=(12, 8))
plt.plot(x_values1, y_means1, marker='o', linestyle='-', color='b', label='Active learning Average F1-score')
plt.fill_between(x_values1, np.array(y_means1) - np.array(y_stds1)/2, np.array(y_means1) + np.array(y_stds1)/2, color='b', alpha=0.2)

plt.plot(x_values1, y_means, marker='o', linestyle='-', color='orange', label='Random Choosing Average F1-score')
plt.fill_between(x_values1, np.array(y_means) - np.array(y_stds)/2, np.array(y_means) + np.array(y_stds)/2, color='orange', alpha=0.2)

plt.xlabel('The size of training dataset',size=18,fontweight='bold')
plt.ylabel('Avg F1-score',size=18,fontweight='bold')
plt.title('Average F1-score ± (Std Dev)/2 Across 8 Experiments (RBC) ',size=18,fontweight='bold')
plt.xticks([20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80])
plt.grid(True)
plt.legend()
# 保存图片，格式可以是 'png', 'pdf', 'svg' 等
plt.savefig("RBC_Average F1-score.png", dpi=800, bbox_inches='tight')
plt.show()