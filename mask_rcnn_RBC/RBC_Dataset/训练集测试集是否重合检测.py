import os

# 定义两个文件夹的路径
train_dir = r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\5-cell-data\11\improve'
val_dir = r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\5-cell-data\11\train'

# 获取文件夹中的所有 .jpg 文件名
train_files = {f for f in os.listdir(train_dir) if f.endswith('.jpg')}
val_files = {f for f in os.listdir(val_dir) if f.endswith('.jpg')}

# 查找重合的文件名
overlap_files = train_files & val_files

# 输出结果
if overlap_files:
    print(f"有重合的图片，共 {len(overlap_files)} 个：")
    print("\n".join(overlap_files))
else:
    print("没有重合的图片。")
