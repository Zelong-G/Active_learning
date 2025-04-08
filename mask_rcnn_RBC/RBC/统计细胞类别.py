import os
from collections import Counter

# 设置数据集所在的文件夹路径
dataset_path = r"D:\work\Active_learning\mask_rcnn\RBC Dataset\COCO_Annotations_png"  # 替换为你的数据集路径

# 统计类别的集合
categories = []

# 遍历文件夹中的文件
for filename in os.listdir(dataset_path):
    if filename.endswith(".png"):  # 确保是图片文件
        parts = filename.split("_")
        if len(parts) >= 3:
            category = parts[1]  # 获取类别部分
            categories.append(category)

# 统计类别数量
category_counts = Counter(categories)

# 输出类别数和类别分布
print(f"总类别数: {len(category_counts)}")
print("类别分布:")
for category, count in category_counts.items():
    print(f"{category}: {count}")
