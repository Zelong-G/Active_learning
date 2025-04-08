import os
import random
import shutil






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


import os
import random
import shutil

def prepare_target_folder(base_target_folder):
    folder_index = 1
    while True:
        existing_folder = f"{base_target_folder}_com{folder_index:02}"
        if not os.path.exists(existing_folder):
            break
        folder_index += 1

    if os.path.exists(base_target_folder):
        os.rename(base_target_folder, existing_folder)
        print(f"原来的文件夹已重命名为: {existing_folder}")

    os.makedirs(base_target_folder, exist_ok=True)
    print(f"新的目标文件夹为: {base_target_folder}")

    train2017_00_folder = f"{base_target_folder}_00"
    if os.path.exists(train2017_00_folder):
        for root, _, files in os.walk(train2017_00_folder):
            for file in files:
                source_path = os.path.join(root, file)
                target_path = os.path.join(base_target_folder, file)
                shutil.copy(source_path, target_path)

    return base_target_folder

def group_files_by_category(source_image_folder):
    category_files = {}
    for filename in os.listdir(source_image_folder):
        if filename.endswith(".jpg"):
            category = filename.split("_")[0]
            category_files.setdefault(category, []).append(filename)
    return category_files

def select_files(category_files, num_total):
    num_per_category = num_total // len(category_files)
    remaining_to_select = num_total % len(category_files)
    selected_files = []

    for category, files in category_files.items():
        if len(files) <= num_per_category:
            print(f"类别 {category} 图片不足 {num_per_category} 张，仅选择 {len(files)} 张。")
            selected_files.extend(files)
        else:
            sampled_files = random.sample(files, num_per_category)
            selected_files.extend(sampled_files)
            category_files[category] = [file for file in files if file not in sampled_files]

    if remaining_to_select > 0:
        print('随机选择总数量不是4的倍数，进行候补操作')
        all_remaining_files = [file for files in category_files.values() for file in files]
        additional_files = random.sample(all_remaining_files, remaining_to_select)

        selected_files.extend(additional_files)
        print(selected_files)
    return selected_files

def select_random_files(source_image_folder, num_total):
    """
    直接在整个文件夹中随机抽取指定数量的图片。
    """
    all_files = [file for file in os.listdir(source_image_folder) if file.endswith(".jpg")]
    if len(all_files) < num_total:
        print(f"警告：文件夹中的图片总数少于 {num_total}，将选择全部 {len(all_files)} 张。")
        return all_files

    return random.sample(all_files, num_total)

def copy_selected_files(selected_files, source_image_folder, target_folder):
    for file in selected_files:
        image_source_path = os.path.join(source_image_folder, file)
        image_target_path = os.path.join(target_folder, file)
        shutil.copy(image_source_path, image_target_path)




# source_image_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell\improve"
# base_target_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\randomchoose\train2017"
# num_total = 100
#
# target_folder = prepare_target_folder(base_target_folder)
# category_files = group_files_by_category(source_image_folder)
# selected_files = select_files(category_files, num_total)
# copy_selected_files(selected_files, source_image_folder, target_folder)
#
# print("图片和标签选择及复制完成！")
#
#
#
#
# # 示例调用
# train2017_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\randomchoose\train2017"
# trainlabel_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\masks"
# output_json = r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\randomchoose\annotations\instances_train2017.json'
# categories = {
#     "LYT": 1,
#     "MON": 2,
#     "MYO": 3,
#     "NGS": 4,
# }
# create_coco_annotations(train2017_folder, trainlabel_folder, output_json, categories)
