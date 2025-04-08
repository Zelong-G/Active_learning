import os
import time
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs

def create_model_analysis(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model

def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def process_folder(folder_path, model, device, category_index, box_thresh, results_csv_path):
    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    results_list = []

    data_transform = transforms.Compose([transforms.ToTensor()])

    for idx, img_file in enumerate(img_files, 1):
        print(f'Processing image {idx}/{len(img_files)}: {img_file}')
        img_path = os.path.join(folder_path, img_file)
        original_img = Image.open(img_path).convert('RGB')

        img = data_transform(original_img).unsqueeze(0).to(device)

        with torch.no_grad():
            img_height, img_width = img.shape[-2:]

            # Warming up the model on a dummy input
            if idx == 1:  # Only for the first image
                dummy_input = torch.zeros((1, 3, img_height, img_width), device=device)
                model(dummy_input)

            results = []
            for _ in range(10):
                predictions = model(img)[0]

                predict_boxes = predictions["boxes"].cpu().numpy()
                predict_classes = predictions["labels"].cpu().numpy()
                predict_scores = predictions["scores"].cpu().numpy()
                predict_mask = predictions["masks"].cpu().numpy()
                predict_mask = np.squeeze(predict_mask, axis=1)
                binary_mask = (predict_mask >= 0.5).astype(np.uint8)

                results.append({
                    "masks": binary_mask,
                    "rois": predict_boxes,
                    "class_ids": predict_classes,
                    "probs": predict_scores
                })

            if results:  # 检测到目标时
                from my_uncertainty_analyzer import calculate_uncertainty
                seg_mean, seg_uncertainties, det_mean, det_uncertainties, cls_mean, cls_uncertainties = calculate_uncertainty(
                    results=results,
                    config=None,
                    category_names=category_index,
                    rare_list=None,
                    rare_thresh=0.5
                )
            else:  # 未检测到目标时
                seg_mean, det_mean, cls_mean = 1, 1, 1

            # 确保值不为空或空字符串
            def safe_assign(value, default=0.8):
                return default if value in [None, '', 'NaN'] else value

            seg_mean = safe_assign(seg_mean)
            det_mean = safe_assign(det_mean)
            cls_mean = safe_assign(cls_mean)

            results_list.append({
                "file_name": img_file,
                "seg_mean": seg_mean,
                "det_mean": det_mean,
                "cls_mean": cls_mean
            })
            # print("seg_mean:",seg_mean)
            # print("cls_mean:", cls_mean)
    pd.DataFrame(results_list).to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

def unce_anal_sta(weights_path,results_csv_path,folder_path):
    import time

    start_time = time.time()  # 记录开始时间
    num_classes = 8
    box_thresh = 0.2
    # weights_path = r"D:\\work\\Active_learning\\mask_rcnn\\run_result\\active\\10\\best.pth"
    # results_csv_path = 'results10.csv'
    # folder_path = r"D:\\work\\Active_learning\\mask_rcnn\\WBC_Dataset\\active-learning-4-cell0\\improve"
    label_json_path = '.\coco91_indices.json'


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    model = create_model_analysis(num_classes=num_classes + 1, box_thresh=box_thresh)

    assert os.path.exists(weights_path), f"{weights_path} file does not exist."
    weights_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(weights_dict.get("model", weights_dict))
    model.to(device).eval()

    assert os.path.exists(label_json_path), f"JSON file {label_json_path} does not exist."
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    process_folder(folder_path, model, device, category_index, box_thresh, results_csv_path)
    end_time = time.time()  # 记录结束时间

    elapsed_time = end_time - start_time  # 计算执行时间
    print(f"代码执行时间: {elapsed_time:.4f} 秒")



# weights_path = r"D:\\work\\Active_learning\\mask_rcnn\\run_result\\active\\10\\best.pth"
# results_csv_path = 'results10.csv'
# unce_anal_sta(weights_path,results_csv_path)

