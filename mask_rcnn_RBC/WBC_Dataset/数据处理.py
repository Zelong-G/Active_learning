import os
import random
import shutil
from collections import defaultdict

def split_cell_images(data_dir, label_dir, output_dir):
    """
    将细胞图片数据分为四个数据集：train, valid, test, improve，并分别保存对应的标签（JPG 格式）。

    参数：
        data_dir (str): 包含细胞图片数据的文件夹路径。
        label_dir (str): 包含标签文件的文件夹路径。
        output_dir (str): 输出数据集的文件夹路径。
    """
    # 检查输出路径是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 数据集文件夹名称
    datasets = ["train2017", "val2017", "test2017", "improve"]

    # 为每个数据集和标签创建文件夹
    for dataset in datasets:
        os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)
        # os.makedirs(os.path.join(output_dir, f"{dataset}label"), exist_ok=True)

    # 按类别分组文件
    files_by_category = defaultdict(list)

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".jpg"):
            category = file_name.split("_")[0]  # 获取类别名称
            files_by_category[category].append(file_name)

    # 确保每类文件随机排序
    for category, files in files_by_category.items():
        random.shuffle(files)

    # 创建 train, valid, test 三个数据集
    dataset_counts = {"train2017": 100, "val2017": 160, "test2017": 160}
    dataset_files = {dataset: [] for dataset in datasets[:3]}  # 仅前三个

    for dataset, count in dataset_counts.items():
        eos_count = 0  # 每个数据集中 EOS 类选 12 个
        other_count = (count - eos_count) // 4  # 其余类每类选取数量

        # 处理 EOS 类
        eos_files = files_by_category["EOS"][:eos_count]
        dataset_files[dataset].extend(eos_files)
        files_by_category["EOS"] = files_by_category["EOS"][eos_count:]

        # 处理其他类
        for category, files in files_by_category.items():
            if category != "EOS":
                selected_files = files[:other_count]
                dataset_files[dataset].extend(selected_files)
                files_by_category[category] = files[other_count:]

    # 保存前三个数据集及其标签
    for dataset, files in dataset_files.items():
        for file_name in files:
            src_image = os.path.join(data_dir, file_name)
            # src_label = os.path.join(label_dir, file_name)  # 标签为 JPG 格式

            dst_image = os.path.join(output_dir, dataset, file_name)
            # dst_label = os.path.join(output_dir, f"{dataset}label", file_name)

            shutil.copy(src_image, dst_image)
            # if os.path.exists(src_label):
            #     shutil.copy(src_label, dst_label)

    # 将剩余数据保存到 improve 数据集中及其标签
    for category, files in files_by_category.items():
        for file_name in files:
            src_image = os.path.join(data_dir, file_name)
            # src_label = os.path.join(label_dir, file_name)  # 标签为 JPG 格式

            dst_image = os.path.join(output_dir, "improve", file_name)
            # dst_label = os.path.join(output_dir, "improvelabel", file_name)

            shutil.copy(src_image, dst_image)
            # if os.path.exists(src_label):
            #     shutil.copy(src_label, dst_label)

    print("数据及标签划分完成！")


import os
import json
import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

def create_coco_annotations(image_dir, mask_dir, output_json, categories):
    """
    根据指定的图片文件夹和掩码文件夹生成 COCO 格式的标签文件。

    :param image_dir: 图片文件夹路径
    :param mask_dir: 掩码文件夹路径
    :param output_json: 输出的 COCO 格式 JSON 文件路径
    :param categories: 类别字典 {"类别名称": 类别ID}
    """
    # 初始化 COCO 数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": id, "name": name, "supercategory": "none"} for name, id in categories.items()]
    }

    image_id = 0
    annotation_id = 0

    # 遍历所有图片
    for image_name in os.listdir(image_dir):
        if image_name.endswith('.jpg'):
            # 文件路径
            img_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name)

            # 提取类别和编号
            label = image_name.split('_')[0]
            category_id = categories.get(label)
            if category_id is None:
                print(f"警告: 未知类别 {label}，跳过 {image_name}")
                continue

            # 读取图片和掩码
            img = Image.open(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # 二值化掩码
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # 图像尺寸
            width, height = img.size

            # 添加图像信息
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height
            })

            # 使用 Pycocotools 提取掩码信息
            binary_mask = binary_mask.astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 保留最大面积的对象
            max_area = 0
            max_segmentation = None
            max_bbox = None

            for contour in contours:
                # 计算轮廓对应的掩码
                temp_mask = np.zeros_like(binary_mask, dtype=np.uint8)
                cv2.drawContours(temp_mask, [contour], -1, 1, thickness=-1)
                temp_rle = maskUtils.encode(np.asfortranarray(temp_mask))
                temp_area = maskUtils.area(temp_rle)
                temp_bbox = maskUtils.toBbox(temp_rle).tolist()

                if temp_area > max_area:
                    max_area = temp_area
                    max_segmentation = contour.flatten().tolist()
                    max_bbox = temp_bbox

            if max_segmentation is None:
                continue

            # 添加注释
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [max_segmentation],
                "area": float(max_area),
                "bbox": max_bbox,
                "iscrowd": 0
            })

            # 更新计数器
            image_id += 1
            annotation_id += 1

    # 保存 JSON 文件
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"COCO 标签文件已保存为 {output_json}")

def process_multiple_datasets(base_dir, datasets, output_dir, categories):
    """
    为多个数据集生成 COCO 格式的标签文件。

    :param base_dir: 基础文件夹路径
    :param datasets: 数据集列表，例如 ["train", "valid", "test", "improve"]
    :param output_dir: 输出 JSON 文件的存放文件夹
    :param categories: 类别字典 {"类别名称": 类别ID}
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dataset in datasets:
        image_dir = os.path.join(base_dir, dataset)
        mask_dir = r'.\WBC_Dataset\masks'
        output_json = os.path.join(output_dir, f"instances_{dataset}.json")

        print(f"正在处理数据集: {dataset}")
        create_coco_annotations(image_dir, mask_dir, output_json, categories)

# 示例调用

# data_dir = r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\4-cell-data'
# output_dir =r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0'
# label_dir = r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\masks'
# split_cell_images(data_dir, label_dir, output_dir)
#
#
# base_dir = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0"
# datasets = ["train2017", "val2017", "test2017", "improve"]
# output_dir = os.path.join(base_dir, "annotations")
# categories = {
#     "LYT": 1,
#     "MON": 2,
#     "MYO": 3,
#     "NGS": 4,
# }
#
# process_multiple_datasets(base_dir, datasets, output_dir, categories)




