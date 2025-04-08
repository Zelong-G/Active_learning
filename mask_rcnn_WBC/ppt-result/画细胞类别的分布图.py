import os
import collections
import matplotlib.pyplot as plt
import seaborn as sns

# 数据集文件夹路径（请替换为你的实际路径）
dataset_path = "../WBC_Dataset/all-bk"

# 统计每个类别的数量
category_counts = collections.Counter()

# 遍历数据集文件夹
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg"):
        category = filename.split("_")[0]
        category_counts[category] += 1

# 转换为可视化数据
categories = list(category_counts.keys())
counts = list(category_counts.values())

# 设置 Seaborn 风格
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# 使用颜色渐变的条形图
colors = sns.color_palette("viridis", len(categories))
bars = plt.bar(categories, counts, color=colors, edgecolor="black")

# 添加数据标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, str(yval),
             ha="center", va="bottom", fontsize=12, fontweight="bold")

# 图表美化
# plt.xlabel("Type of white blood cells", fontsize=14, fontweight="bold")
plt.ylabel("number of cells", fontsize=18, fontweight="bold")
plt.title("Type of white blood cells", fontsize=18, fontweight="bold")
plt.xticks(rotation=0, fontsize=24, fontweight="bold")
plt.yticks(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# 去除顶部和右侧边框
sns.despine()
# 保存图片，格式可以是 'png', 'pdf', 'svg' 等
plt.savefig("WBC_class.png", dpi=800, bbox_inches='tight')

# 展示图表
plt.show()
