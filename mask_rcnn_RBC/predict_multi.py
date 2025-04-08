import os
import time
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs

def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    num_classes = 8  # 不包含背景
    box_thresh = 0.5
    weights_path = r"D:\work\Active_learning\mask_rcnn\run_result\完整数据-噪音-40epoch-无dropout\model_38.pth"
    img_folder = r"D:\work\Active_learning\mask_rcnn\TBZ"
    output_folder = r"D:\work\Active_learning\mask_rcnn\inference_result\8"
    label_json_path = './coco91_indices.json'

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file does not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} does not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # 读取文件夹中的所有图片
    assert os.path.exists(img_folder), f"{img_folder} does not exist."
    img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('jpg', 'jpeg', 'png', 'tif'))]

    if not img_files:
        print("No images found in the folder.")
        return

    data_transform = transforms.Compose([transforms.ToTensor()])

    model.eval()  # 进入验证模式
    with torch.no_grad():
        for img_name in img_files:
            img_path = os.path.join(img_folder, img_name)
            original_img = Image.open(img_path).convert('RGB')

            # from pil image to tensor, do not normalize image
            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)  # expand batch dimension

            # 推理
            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print(f"{img_name}: inference+NMS time: {t_end - t_start}")

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_mask = predictions["masks"].to("cpu").numpy()
            predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]
            # 将 predict_mask 进行二值化
            binary_mask = (predict_mask > 0.5).astype(np.uint8)  # 阈值为 0.5，转换为 0 或 1

            if len(predict_boxes) == 0:
                print(f"{img_name}: No objects detected!")
                continue

            # 绘制预测结果
            plot_img = draw_objs(original_img,
                                 boxes=predict_boxes,
                                 classes=predict_classes,
                                 scores=predict_scores,
                                 masks=predict_mask,
                                 category_index=category_index,
                                 line_thickness=3,
                                 font='arial.ttf',
                                 font_size=20)

            # 保存预测结果
            output_path = os.path.join(output_folder, img_name)
            plot_img.save(output_path)

            binary_mask=remove_small_regions_by_connected_components(binary_mask, min_area=250)
            # # 只保存细胞 背景透明的 png
            # for i, mask in enumerate(binary_mask):
            #     transparent_png = generate_transparent_png(original_img, mask)
            #     output_path = os.path.join(output_folder,img_name+f"segmentation_result_{i}.png")
            #     transparent_png.save(output_path)
            #     break

            # 示例代码
            for g, mask1 in enumerate(binary_mask):
                black_white_img = generate_black_white_mask(mask1)
                output_path = os.path.join(output_folder,img_name+f"mask_result_{g}.png")
                black_white_img.save(output_path)
                break
    print("All images processed and results saved.")


from PIL import Image
import numpy as np


def generate_transparent_png(original_img, mask):
    """
    根据分割 mask 将背景设为透明，并生成 PNG 图像。

    :param original_img: 原始图片（PIL Image 格式）
    :param mask: 二维 mask 数组，值为 0 或 1，1 表示前景区域
    :return: 带透明背景的 PNG 图像
    """
    # 确保原始图片和 mask 尺寸一致
    original_img = original_img.resize((mask.shape[1], mask.shape[0]))

    # 将原始图片转为 RGBA 模式
    original_img = original_img.convert("RGBA")
    original_array = np.array(original_img)

    # 创建透明背景
    transparent_array = np.zeros_like(original_array, dtype=np.uint8)

    # 将 mask 为 1 的区域保留原图像素，其他区域透明
    transparent_array[..., :3] = original_array[..., :3]  # RGB 通道
    transparent_array[..., 3] = (mask * 255).astype(np.uint8)  # Alpha 通道

    # 转换为 PIL 图像
    transparent_img = Image.fromarray(transparent_array, mode="RGBA")
    return transparent_img


def generate_black_white_mask(mask):
    """
    根据分割 mask 生成黑白图片，背景为黑色，mask 区域为白色。

    :param mask: 二维 mask 数组，值为 0 或 1，1 表示前景区域
    :return: 黑白图片（PIL Image 格式）
    """
    # 将 mask 的值从 0/1 转换为 0/255
    black_white_mask = (mask * 255).astype(np.uint8)

    # 转换为 PIL Image 格式
    black_white_image = Image.fromarray(black_white_mask, mode="L")
    return black_white_image


import numpy as np
import cv2


def remove_small_regions_by_connected_components(binary_mask, min_area=10000):
    """
    使用连通域分析去除小区域的 mask。

    Args:
        binary_mask (np.ndarray): 输入的二值化 mask，形状为 [batch, h, w]。
        min_area (int): 保留的最小连通域面积，小于该值的区域会被移除。

    Returns:
        np.ndarray: 去除了小区域的 mask，形状与输入相同。
    """
    cleaned_mask = np.zeros_like(binary_mask)
    for i in range(binary_mask.shape[0]):  # 遍历 batch
        mask = binary_mask[i]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, num_labels):  # 从1开始，0是背景
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned_mask[i][labels == label] = 1
    return cleaned_mask


import cv2


def remove_small_regions_by_morphology(binary_mask, kernel_size=(3, 3)):
    """
    使用形态学开操作去除小区域的 mask。

    Args:
        binary_mask (np.ndarray): 输入的二值化 mask，形状为 [batch, h, w]。
        kernel_size (tuple): 形态学操作的内核大小，默认为 (3, 3)。

    Returns:
        np.ndarray: 去除了小区域的 mask，形状与输入相同。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    cleaned_mask = np.zeros_like(binary_mask)
    for i in range(binary_mask.shape[0]):  # 遍历 batch
        mask = binary_mask[i]
        cleaned_mask[i] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cleaned_mask


if __name__ == '__main__':
    main()
