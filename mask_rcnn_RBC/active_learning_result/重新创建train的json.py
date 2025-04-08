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

# 示例调用
train2017_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\randomchoose\train2017"
trainlabel_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\masks"
output_json = r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\randomchoose\annotations\instances_train2017.json'
categories = {
    "LYT": 1,
    "MON": 2,
    "MYO": 3,
    "NGS": 4,
}
create_coco_annotations(train2017_folder, trainlabel_folder, output_json, categories)
