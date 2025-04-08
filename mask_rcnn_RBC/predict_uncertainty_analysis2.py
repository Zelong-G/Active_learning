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
import cv2

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
    weights_path = "./RBC_Dataset/active_learning01/result/1/best.pth"
    img_path = "./RBC_Dataset/active_learning01/val2017/fr frame 19.png"
    label_json_path = "./coco91_indices.json"

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)


    model.eval()  # 进入验证模式
    # 在主函数中引入结果存储
    results = []
    # 初始化分割结果累加器
    cumulative_mask = None
    for i in range(10):
        # load image
        assert os.path.exists(img_path), f"{img_path} does not exits."
        original_img = Image.open(img_path).convert('RGB')

        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)

            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_mask = predictions["masks"].to("cpu").numpy()
            predict_mask = np.squeeze(predict_mask, axis=1)
            # 二值化 mask (阈值为0.5)
            binary_mask = (predict_mask >= 0.5).astype(np.uint8)
            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
                return

            # 将当前结果存储到 results 列表
            results.append({
                "masks": binary_mask,
                "rois": predict_boxes,
                "class_ids": predict_classes,
                "probs": predict_scores
            })
            flag =0
            for mask in binary_mask:
                ####提取实例
                # 创建一个透明背景的图像 (RGBA)
                transparent_image = np.zeros((img_height, img_width, 4), dtype=np.uint8)
                # 将原始图片的目标区域拷贝到透明背景图像中
                original_img_np = np.array(original_img)
                for c in range(3):  # 处理 R、G、B 通道
                    transparent_image[:, :, c] = original_img_np[:, :, c] * mask

                # 设置 Alpha 通道
                transparent_image[:, :, 3] = mask * 255  # Mask 中的 1 对应完全不透明（255），0 对应完全透明（0）
                # 保存为 PNG 文件
                output_path = "instance_mask_output"+str(flag)+".png"
                flag+=1
                Image.fromarray(transparent_image).save(output_path)



            # 二值化 mask (阈值为0.5)
            binary_mask = (predict_mask >= 0.5).astype(np.uint8)

            # 初始化累加器
            if cumulative_mask is None:
                cumulative_mask = np.zeros_like(binary_mask[0], dtype=np.uint8)

            # 累加每个实例的 mask
            for instance_mask in binary_mask:
                cumulative_mask += instance_mask




            plot_img = draw_objs(original_img,
                                 boxes=predict_boxes,
                                 classes=predict_classes,
                                 scores=predict_scores,
                                 masks=predict_mask,
                                 category_index=category_index,
                                 line_thickness=2,
                                 font='arial.ttf',
                                 font_size=15)
            # 去掉坐标轴
            plt.axis('off')  # 隐藏坐标轴

            # 去掉多余的白边
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 调整图像边距
            plt.imshow(plot_img)
            plt.show()
            plt.close()
    # 调用不确定性分析
    config = None  # 如果需要额外配置，可以创建一个配置对象
    rare_list = None  # 如果有稀有类别的定义，可以传入类别索引
    rare_thresh = 0.5  # 稀有实例的概率阈值
    from my_uncertainty_analyzer import calculate_uncertainty
    # 调用不确定性分析函数
    seg_mean, seg_uncertainties, det_mean, det_uncertainties, cls_mean, cls_uncertainties = calculate_uncertainty(
        results=results,
        config=config,
        category_names=category_index,
        rare_list=rare_list,
        rare_thresh=rare_thresh
            )

    # 可视化累计 mask
    visualize_cumulative_mask(original_img, cumulative_mask)



def visualize_cumulative_mask(original_img, cumulative_mask):
    """对累计的 mask 进行分段着色并叠加到原图"""
    cumulative_mask = np.clip(cumulative_mask, 0, 10)  # 确保值在 0 到 10 范围内

    # 定义颜色映射：0为透明，1-3为浅蓝，4-6为中蓝，7-10为深蓝
    color_map = np.zeros((*cumulative_mask.shape, 3), dtype=np.uint8)
    color_map[cumulative_mask == 0] = [0, 0, 0]  # 背景透明
    color_map[(cumulative_mask >= 1) & (cumulative_mask <= 4)] = [173, 216, 230]  # 浅蓝色
    color_map[(cumulative_mask >= 5) & (cumulative_mask <= 8)] = [100, 149, 237]  # 中蓝色
    color_map[(cumulative_mask >= 9) & (cumulative_mask <= 10)] = [25, 25, 112]  # 深蓝色

    # 创建带透明度的叠加图像
    alpha = 0.8  # 透明度因子
    original_img = np.array(original_img)
    overlay_img = original_img.copy()
    for i in range(1, 11):  # 逐级绘制不同级别的 mask
        mask_level = (cumulative_mask == i)
        overlay_img[mask_level] = (
                alpha * color_map[mask_level] + (1 - alpha) * overlay_img[mask_level]
        ).astype(np.uint8)


    # 显示叠加结果
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_img)
    plt.axis('off')
    plt.title("Cumulative Mask Visualization")
    plt.show()


if __name__ == '__main__':
    main()

