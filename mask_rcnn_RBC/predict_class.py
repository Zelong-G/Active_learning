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

def compute_accuracy(pred_classes, gt_classes):
    """ 计算分类准确度 """
    correct = sum(1 for pred, gt in zip(pred_classes, gt_classes) if pred == gt)
    accuracy = correct / len(gt_classes) if gt_classes else 0
    return accuracy

def main():
    num_classes = 8  # 不包含背景
    box_thresh = 0.5
    weights_path = "./run_result/active/best.pth"
    img_path = "./RBC Dataset/dataset01/val2017/fr frame 18.png"
    label_json_path = "./coco91_indices.json"
    gt_label_path = "./RBC Dataset/dataset01/annotations/instances_val2017.json"  # 真实标签文件

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)
    assert os.path.exists(weights_path), "{} file does not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.backbone.fpn.set_inference_mode(False)
    model.to(device)

    assert os.path.exists(label_json_path), "json file {} does not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    assert os.path.exists(gt_label_path), "Ground truth file {} does not exist.".format(gt_label_path)
    with open(gt_label_path, 'r') as gt_file:
        ground_truth_data = json.load(gt_file)

    assert os.path.exists(img_path), f"{img_path} does not exist."
    original_img = Image.open(img_path).convert('RGB')
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
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

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")
            return

        gt_classes = ground_truth_data.get(os.path.basename(img_path), [])
        accuracy = compute_accuracy(predict_classes, gt_classes)
        print(f"分类准确度: {accuracy * 100:.2f}%")

        plot_img = draw_objs(original_img,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             category_index=category_index,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()
        plot_img.save("test_result.jpg")

if __name__ == '__main__':
    main()
