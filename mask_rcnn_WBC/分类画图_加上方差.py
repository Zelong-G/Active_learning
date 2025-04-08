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
                    if len(values) >= 5:
                        return float(values[4])  # 第五个值是 f1-score
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

# 基础路径
base_path = "./run_result"
experiment_folders = [f"active{i}" for i in range(5, 8)]  # 5 到 7

# 存储 y 轴数据
folder_f1_scores_random = {i: [] for i in range(2, 7)}  # 存储每个 folder 2-6 的 f1-score
folder_f1_scores_active = {i: [] for i in range(1, 7)}  # 存储每个 folder 1-6 的 f1-score

# 遍历所有实验文件夹（随机选择）
for exp in experiment_folders:
    result_path = os.path.join(base_path, exp)
    for i in range(2, 7):
        file_path = os.path.join(result_path, str(i)+'random_com', "classification_results.txt")
        if os.path.exists(file_path):
            f1_score = extract_macro_avg_f1(file_path)
            if f1_score is not None:
                folder_f1_scores_random[i].append(f1_score)

# 遍历所有实验文件夹（主动学习）
for exp in experiment_folders:
    result_path = os.path.join(base_path, exp)
    for i in range(1, 7):
        file_path = os.path.join(result_path, str(i), "classification_results.txt")
        if os.path.exists(file_path):
            f1_score = extract_macro_avg_f1(file_path)
            if f1_score is not None:
                folder_f1_scores_active[i].append(f1_score)

# 计算平均值和标准差
x_values = np.array([120, 140, 160, 180, 200])
x_values1 = np.array([100, 120, 140, 160, 180, 200])

y_values_random = np.array([np.mean(folder_f1_scores_random[i]) if folder_f1_scores_random[i] else 0 for i in range(2, 7)])
std_random = np.array([np.std(folder_f1_scores_random[i]) if folder_f1_scores_random[i] else 0 for i in range(2, 7)])
y_values_random = np.insert(y_values_random, 0, np.mean(folder_f1_scores_active[1]) if folder_f1_scores_active[1] else 0)
std_random = np.insert(std_random, 0, np.std(folder_f1_scores_active[1]) if folder_f1_scores_active[1] else 0)

y_values_active = np.array([np.mean(folder_f1_scores_active[i]) if folder_f1_scores_active[i] else 0 for i in range(1, 7)])
std_active = np.array([np.std(folder_f1_scores_active[i]) if folder_f1_scores_active[i] else 0 for i in range(1, 7)])

# 绘制图表
plt.figure(figsize=(12, 8))
plt.plot(x_values1, y_values_active, marker='o', linestyle='-', color='b', label='Active learning Average F1-score')
plt.fill_between(x_values1, y_values_active - std_active/2, y_values_active + std_active/2, color='b', alpha=0.2)

plt.plot(x_values1, y_values_random, marker='o', linestyle='-', color='orange', label='Random Average F1-score')
plt.fill_between(x_values1, y_values_random - std_random/2, y_values_random + std_random/2, color='orange', alpha=0.2)

plt.xlabel('The size of training dataset',size=18,fontweight='bold')
plt.ylabel('Avg F1-score',size=18,fontweight='bold')
plt.title('Average F1-score ± (Std Dev)/2 Across 5 Experiments (WBC)',size=18,fontweight='bold')
plt.xticks([100, 120, 140, 160, 180, 200])
plt.grid(True)
plt.legend()
# 保存图片，格式可以是 'png', 'pdf', 'svg' 等
plt.savefig("WBC_Average F1-score.png", dpi=800, bbox_inches='tight')

plt.show()
