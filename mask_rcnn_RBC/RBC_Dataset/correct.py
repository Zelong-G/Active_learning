import os
import shutil

# 设置文件夹路径
image_folder = r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\data4\train2017'
label_folder = r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\masks'
output_folder = r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\data4\train_mask_jpg'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取图片和标注文件列表
image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))}
label_files = {os.path.splitext(f)[0]: f for f in os.listdir(label_folder) if f.endswith(('.jpg', '.json'))}

# 筛选有标注的图片
matched_files = set(image_files.keys()) & set(label_files.keys())

# 移动筛选出的图片和标注到输出文件夹
for file in matched_files:
    shutil.copy(os.path.join(image_folder, image_files[file]), os.path.join(output_folder, image_files[file]))
    shutil.copy(os.path.join(label_folder, label_files[file]), os.path.join(output_folder, label_files[file]))

print(f"筛选出 {len(matched_files)} 个带有标签的图片及标注文件，已存储在 '{output_folder}' 文件夹中。")
