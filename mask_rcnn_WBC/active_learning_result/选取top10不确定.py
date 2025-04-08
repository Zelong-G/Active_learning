import csv
import os
import shutil

def process_and_copy_files(csv_file, improve_folder, train2017_folder):
    """
    从 CSV 文件中读取数据，找到最后三列和最小的 10 个文件，
    将其对应的图片放在原 train2017 文件夹，并将原 train2017 文件夹备份。

    参数：
        csv_file (str): CSV 文件路径。
        improve_folder (str): 原始图片所在文件夹路径。
        train2017_folder (str): 原始 train2017 文件夹路径。
        improvelabel_folder (str): 原始标签文件夹路径。
        trainlabel_folder (str): 目标标签文件夹路径。
    """
    # 读取 CSV 文件并计算每行最后三个值的和
    data = []
    try:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # 跳过标题行
            for row in reader:
                values_sum = sum(map(float, row[1:]))  # 转换最后三列为浮点数并求和
                data.append((row[0], values_sum))  # 保存文件名和求和值
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # 找到求和值最低的 10 行数据
    lowest_files = sorted(data, key=lambda x: x[1])[:20]
    print(f"Lowest files: {lowest_files}")
    lowest_files = [str(f[0]) for f in lowest_files]

    # 创建 train2017 的备份文件夹
    train2017_copy_folder = train2017_folder
    suffix = 0
    while True:
        new_train_folder = f"{train2017_copy_folder}_{suffix:02d}"
        if not os.path.exists(new_train_folder):
            shutil.copytree(train2017_folder, new_train_folder)  # 复制 train2017 文件夹
            print(f"Backed up {train2017_folder} to {new_train_folder}")
            break
        suffix += 1

    # 将选定的图片移动到原 train2017 文件夹
    for file_name in lowest_files:
        source_image = os.path.join(improve_folder, file_name)
        target_image = os.path.join(train2017_folder, file_name)
        if os.path.exists(source_image):
            shutil.move(source_image, target_image)
            print(f"Moved {source_image} to {target_image}")
        else:
            print(f"Image not found: {source_image}")



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

def create_coco_annotations(image_dir, mask_dir, output_json, categories):
    """
    根据指定的图片文件夹和掩码文件夹生成 COCO 格式的标签文件。

    :param image_dir: 图片文件夹路径
    :param mask_dir: 掩码文件夹路径
    :param output_json: 输出的 COCO 格式 JSON 文件路径
    :param categories: 类别字典 {"类别名称": 类别ID}
    """
    # 备份 JSON 文件
    backup_json_file(output_json)

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


# # 示例调用
# csv_file = r'D:\work\Active_learning\mask_rcnn\active-learning-result\results10.csv'
# improve_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\improve_200"
# train2017_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\train2017"
#
# process_and_copy_files(csv_file, improve_folder, train2017_folder)
#
#
# # 示例调用
# train2017_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\train2017"
# trainlabel_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\masks"
# output_json = r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\annotations\instances_train2017.json'
# categories = {
#     "LYT": 1,
#     "MON": 2,
#     "MYO": 3,
#     "NGS": 4,
# }
# create_coco_annotations(train2017_folder, trainlabel_folder, output_json, categories)
