import os
import shutil
import random
from collections import defaultdict

# 数据路径和目标路径
data_dir = r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\all-bk'  # 替换为图片数据集路径
output_dir = 'divide'  # 替换为输出路径

# 创建输出目录
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 类别和划分比例配置
split_ratios = {
    'EOS': {'train': 0.7, 'test': 0.3},
    'LYT': {'train': 0.6, 'test': 0.4},
    'MON': {'train': 0.7, 'test': 0.3},
    'MYO': {'train': 0.7, 'test': 0.3},
    'NGS': {'train': 0.6, 'test': 0.4},

    'BAS': {'train': 0.7, 'test': 0.3},
    'EBO': {'train': 0.7, 'test': 0.3},
    'NGB': {'train': 0.7, 'test': 0.3},

}

# 收集所有图片文件
files_by_class = defaultdict(list)
for file in os.listdir(data_dir):
    if os.path.isfile(os.path.join(data_dir, file)):
        class_label = file.split('_')[0]  # 根据文件名前缀判断类别
        if class_label in split_ratios:
            files_by_class[class_label].append(file)

# 数据划分
for class_label, files in files_by_class.items():
    random.shuffle(files)
    num_files = len(files)
    train_count = int(num_files * split_ratios[class_label]['train'])
    test_count = int(num_files * split_ratios[class_label]['test'])

    # 分配文件到训练集
    train_files = files[:train_count]
    test_files = files[train_count:train_count + test_count]

    # 确保训练集和测试集不重叠
    assert len(set(train_files) & set(test_files)) == 0, "训练集和测试集存在重复文件！"

    for file in train_files:
        shutil.copy(os.path.join(data_dir, file), os.path.join(train_dir, file))

    for file in test_files:
        shutil.copy(os.path.join(data_dir, file), os.path.join(test_dir, file))

print("数据集划分完成！")
