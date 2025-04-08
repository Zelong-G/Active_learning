import os
import time
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import pandas as pd

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


def process_folder(folder_path, model, device, category_index, box_thresh, results_csv_path):
    # 获取文件夹中所有图片文件
    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    results_list = []
    flag = 1
    for img_file in img_files:
        print('现在处理到第'+str(flag)+'张图片')
        flag+=1
        img_path = os.path.join(folder_path, img_file)
        original_img = Image.open(img_path).convert('RGB')

        # from PIL image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        img = torch.unsqueeze(img, dim=0).to(device)

        with torch.no_grad():
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            results = []
            for _ in range(10):
                predictions = model(img)[0]

                predict_boxes = predictions["boxes"].to("cpu").numpy()
                predict_classes = predictions["labels"].to("cpu").numpy()
                predict_scores = predictions["scores"].to("cpu").numpy()
                predict_mask = predictions["masks"].to("cpu").numpy()

                if len(predict_boxes) > 0:
                    max_score_index = np.argmax(predict_scores)
                    predict_boxes = predict_boxes[max_score_index:max_score_index + 1]
                    predict_classes = predict_classes[max_score_index:max_score_index + 1]
                    predict_scores = predict_scores[max_score_index:max_score_index + 1]
                    predict_mask = predict_mask[max_score_index:max_score_index + 1]

                    predict_mask = np.squeeze(predict_mask, axis=1)
                    binary_mask = (predict_mask >= 0.5).astype(np.uint8)

                    results.append({
                        "masks": binary_mask,
                        "rois": predict_boxes,
                        "class_ids": predict_classes,
                        "probs": predict_scores
                    })

            # 调用不确定性分析
            config = None
            rare_list = None
            rare_thresh = 0.5

            from my_uncertainty_analyzer import calculate_uncertainty
            seg_mean, seg_uncertainties, det_mean, det_uncertainties, cls_mean, cls_uncertainties = calculate_uncertainty(
                results=results,
                config=config,
                category_names=category_index,
                rare_list=rare_list,
                rare_thresh=rare_thresh
            )

            # 保存结果
            results_list.append({
                "file_name": img_file,
                "seg_mean": seg_mean,
                "det_mean": det_mean,
                "cls_mean": cls_mean
            })

    # 保存为 CSV 文件
    df = pd.DataFrame(results_list)
    df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")


def main():
    num_classes = 4
    box_thresh = 0.2
    weights_path = r"D:\work\Active_learning\mask_rcnn\run_result\9\last.pth"
    folder_path = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell\improve"
    label_json_path = './coco91_indices.json'
    results_csv_path = 'active-learning-result/results.csv'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    assert os.path.exists(weights_path), f"{weights_path} file does not exist."
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    assert os.path.exists(label_json_path), f"json file {label_json_path} does not exist."
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    model.eval()

    process_folder(folder_path, model, device, category_index, box_thresh, results_csv_path)


if __name__ == '__main__':
    main()
