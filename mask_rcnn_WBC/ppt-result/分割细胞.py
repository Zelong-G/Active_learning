import json
import os
from PIL import Image, ImageDraw
import numpy as np

# 读取JSON文件
json_file_path = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\data3\annotations\instances_val2017.json"  # 替换为你的JSON文件路径
images_folder = r"D:\work\Active_learning\mask_rcnn\ppt-result\mask"  # 替换为存储细胞图像的文件夹路径
output_folder = r"D:\work\Active_learning\mask_rcnn\ppt-result\分割mask2"  # 输出文件夹路径

os.makedirs(output_folder, exist_ok=True)

# 加载 JSON 文件
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 创建 ID 到文件名的映射
image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}

# 处理每个标注
for annotation in data['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']  # 可用于区分类别
    segmentation = annotation['segmentation'][0]  # 假定单一分割

    # 获取图像文件路径
    if image_id not in image_id_to_filename:
        print(f"跳过：未找到 image_id 为 {image_id} 的文件名映射")
        continue

    image_file_name = image_id_to_filename[image_id]
    image_path = os.path.join(images_folder, image_file_name)

    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"跳过：文件 {image_path} 不存在")
        continue

    # 打开原图并生成透明背景
    with Image.open(image_path).convert("RGBA") as img:
        # 创建一个与图像尺寸相同的透明背景
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)

        # 根据分割点绘制多边形
        polygon = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
        draw.polygon(polygon, fill=255)

        # 将 mask 应用于图像
        mask_np = np.array(mask)
        img_np = np.array(img)

        # 设置透明背景
        img_np[mask_np == 0] = [0, 0, 0, 0]  # 将非细胞部分设置为透明

        # 保存带透明背景的图片
        output_path = os.path.join(output_folder, f"{os.path.splitext(image_file_name)[0]}_mask.png")
        Image.fromarray(img_np).save(output_path)

        print(f"已处理：{image_file_name}，输出到 {output_path}")

print(f"处理完成，所有可用文件已保存到 {output_folder}")
