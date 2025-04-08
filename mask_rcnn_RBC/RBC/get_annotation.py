import os
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils

def split(dataset_path,output_dir):
    import os
    import shutil
    import random

    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train2017", "test2017", "val2017", "improve"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # 获取所有图片
    images = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)  # 打乱顺序

    # 按比例分配
    train_size = 20
    test_size = 50
    val_size = 50

    train_images = images[:train_size]
    test_images = images[train_size:train_size + test_size]
    val_images = images[train_size + test_size:train_size + test_size + val_size]
    remaining_images = images[train_size + test_size + val_size:]

    # 复制图片到对应文件夹
    for img in train_images:
        shutil.copy(os.path.join(dataset_path, img), os.path.join(output_dir, "train2017", img))

    for img in test_images:
        shutil.copy(os.path.join(dataset_path, img), os.path.join(output_dir, "test2017", img))

    for img in val_images:
        shutil.copy(os.path.join(dataset_path, img), os.path.join(output_dir, "val2017", img))

    for img in remaining_images:
        shutil.copy(os.path.join(dataset_path, img), os.path.join(output_dir, "improve", img))

    print(f"数据集划分完成，结果存放在 {output_dir} 目录下！")


def get_mask_files(image_file,MASK_DIR):
    """根据图像文件名获取对应的mask文件"""
    base_name = image_file.rsplit(".", 1)[0]  # 去掉文件后缀
    masks = [f for f in os.listdir(MASK_DIR) if f.startswith(base_name + "_")]
    return masks


def binary_mask_to_polygon(binary_mask):
    """将二值掩码转换为 COCO 格式的多边形 segmentation"""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:  # 至少需要 2 个点
            segmentation.append(contour)
    return segmentation

import os
import json
import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

def backup_json_file(output_json):
    """
    备份 JSON 文件，如果文件存在则依次创建 `instances_train2017_00.json`、`instances_train2017_01.json` 等。

    :param output_json: 原始 JSON 文件路径
    """
    if not os.path.exists(output_json):
        return  # 如果文件不存在，无需备份

    # 获取文件目录和名称
    base_folder = os.path.dirname(output_json)
    base_name = os.path.basename(output_json).split('.')[0]
    suffix = 0

    # 生成新的备份文件名
    while True:
        backup_name = f"{base_name}_{suffix:02d}.json"
        backup_path = os.path.join(base_folder, backup_name)
        if not os.path.exists(backup_path):
            os.rename(output_json, backup_path)  # 重命名为备份文件
            print(f"备份文件已创建：{backup_path}")
            break
        suffix += 1


def create_coco_json(image_dir, output_json,MASK_DIR):
    backup_json_file(output_json)
    categories = {}
    category_id = 1
    annotations = []
    images = []
    image_id = 1
    annotation_id = 1

    image_files = sorted(os.listdir(image_dir))
    for image_file in image_files:
        masks = get_mask_files(image_file,MASK_DIR)
        if not masks:
            continue

        # 添加 Image 记录
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        images.append({
            "id": image_id,
            "file_name": image_file,
            "width": image.shape[1],
            "height": image.shape[0]
        })

        for mask_file in masks:
            # 解析文件名
            parts = mask_file.rsplit("_", 2)  # 分割最后两个 "_"
            if len(parts) < 3:
                continue
            class_name = parts[1]  # 细胞类别

            # 记录类别
            if class_name not in categories:
                categories[class_name] = category_id
                category_id += 1

            # 读取 mask
            mask_path = os.path.join(MASK_DIR, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                continue

            # 计算边界框
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue

            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

            # 计算多边形 segmentation
            segmentation = binary_mask_to_polygon(mask)
            if not segmentation:
                continue

            # 添加 Annotation
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": categories[class_name],
                "segmentation": segmentation,
                "area": int(np.sum(mask > 0)),
                "bbox": bbox,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    # 生成 COCO 格式 JSON
    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": v, "name": k} for k, v in categories.items()]
    }
    # 确保目录存在
    output_dir = os.path.dirname(output_json)  # 获取文件的父目录
    os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在
    # 保存 JSON
    with open(output_json, "w") as f:
        json.dump(coco_json, f, indent=4)

    print(f"COCO 标注文件已生成: {output_json}")

# # 数据集路径
# TRAIN_IMAGE_DIR = "../RBC Dataset/dataset01/train2017"
# TEST_IMAGE_DIR = "../RBC Dataset/dataset01/val2017"
# MASK_DIR = "../RBC Dataset/COCO_Annotations_png"  # 所有标签的文件夹
# OUTPUT_TRAIN_JSON = "../RBC Dataset/dataset01/annotations/instances_train2017.json"
# OUTPUT_TEST_JSON = "../RBC Dataset/dataset01/annotations/instances_val2017.json"
#
# # 生成训练集和测试集的 JSON 文件
# create_coco_json(TRAIN_IMAGE_DIR, OUTPUT_TRAIN_JSON,MASK_DIR)
# create_coco_json(TEST_IMAGE_DIR, OUTPUT_TEST_JSON,MASK_DIR)


