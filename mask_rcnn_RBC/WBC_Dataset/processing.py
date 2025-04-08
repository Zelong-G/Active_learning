import os
import json
import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

# 文件夹路径
image_dir = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\data4\val2017"
mask_dir = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\data4\val_mask_jpg"
output_json = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\data4\annotations\instances_val2017.json"

# 类别映射
categories = {
    "EOS": 1,
    "LYT": 2,
    "MON": 3,
    "MYO": 4,
    "NGS": 5,
    "BAS": 6,
    "EBO": 7,
    "NGB": 8,

}

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
        category_id = categories[label]

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
        rle = maskUtils.encode(np.asfortranarray(binary_mask))
        area = maskUtils.area(rle)
        bbox = maskUtils.toBbox(rle).tolist()

        # 轮廓提取分割信息
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for contour in contours:
            segmentation.append(contour.flatten().tolist())

        # 添加注释
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": float(area),
            "bbox": bbox,
            "iscrowd": 0
        })

        # 更新计数器
        image_id += 1
        annotation_id += 1

# 保存 JSON 文件
with open(output_json, 'w') as f:
    json.dump(coco_data, f, indent=4)

print(f"COCO 标签文件已保存为 {output_json}")
