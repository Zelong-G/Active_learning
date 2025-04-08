import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from my_dataset_coco import CocoDetection
from train_utils import train_eval_utils as utils
import transforms
import torch.utils.data as data
from pycocotools.coco import COCO
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.ops import box_iou
import json


def load_category_names(annotation_file):
    """
    从 COCO 格式的 JSON 文件加载类别名称
    """
    with open(annotation_file, "r") as f:
        coco_data = json.load(f)

    categories = coco_data["categories"]
    category_names = {cat["id"]: cat["name"] for cat in categories}

    return category_names

def create_model_eval(num_classes):
    # 创建模型
    backbone = resnet50_fpn_backbone(pretrain_path="resnet50.pth", trainable_layers=3)
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model


def match_predictions_to_gt(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_threshold=0.5):
    """
    通过 IoU 计算 GT 目标和预测目标的最佳匹配关系
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return [], []  # 没有匹配目标

    # 计算 IoU
    ious = box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes))  # shape: (num_pred, num_gt)

    matched_gt = []
    matched_pred = []

    # 为每个 GT 目标找到 IoU 最大的预测目标
    gt_matched = set()
    pred_matched = set()

    for pred_idx in range(len(pred_boxes)):
        max_iou, gt_idx = torch.max(ious[pred_idx], dim=0)
        if max_iou > iou_threshold and gt_idx.item() not in gt_matched:
            matched_pred.append(pred_labels[pred_idx])
            matched_gt.append(gt_labels[gt_idx])
            gt_matched.add(gt_idx.item())
            pred_matched.add(pred_idx)

    return matched_gt, matched_pred


def eval(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device.type))

    # 读取 COCO 数据集的类别名称
    annotation_file = os.path.join(args.data_path, "annotations/instances_test2017.json")
    category_names = load_category_names(annotation_file)

    # 数据预处理
    data_transform = {
        "test": transforms.Compose([transforms.ToTensor()])
    }

    # 加载测试数据集
    val_dataset = CocoDetection(args.data_path, "test", data_transform["test"])

    # 创建测试数据加载器
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,  # Windows 下先用 0，测试后再改
        collate_fn=val_dataset.collate_fn
    )

    # 创建模型并加载权重
    model = create_model_eval(num_classes=args.num_classes + 1)
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.backbone.fpn.set_inference_mode(False)
    model.to(device)
    model.eval()

    # 评估模型
    det_info, seg_info = utils.evaluate(model, val_data_loader, device=device)

    # 计算分类结果
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, targets in val_data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for target, output in zip(targets, outputs):
                gt_boxes = target["boxes"].cpu().numpy()  # GT 边界框
                gt_labels = target["labels"].cpu().numpy()  # GT 标签

                pred_boxes = output["boxes"].cpu().numpy()  # 预测边界框
                pred_labels = output["labels"].cpu().numpy()  # 预测标签

                # 进行匹配
                matched_gt, matched_pred = match_predictions_to_gt(gt_boxes, gt_labels, pred_boxes, pred_labels)
                # print('matched_gt:', matched_gt)
                # print('matched_pr:', matched_pred)
                # 添加到总的分类列表
                y_true.extend(matched_gt)
                y_pred.extend(matched_pred)

    # 替换类别索引为类别名称
    labels = sorted(category_names.keys())  # 确保类别索引按顺序排列
    target_names = [category_names[label] for label in labels]

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    class_report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)

    # 结果存储路径
    weights_dir = os.path.dirname(args.weights)
    class_results_file = os.path.join(weights_dir, "classification_results.txt")

    # 写入分类结果
    with open(class_results_file, "w") as f:
        f.write("Classification Report:\n")
        f.write(class_report + "\n")

    print(f"Classification results saved to {class_results_file}")


if __name__ == '__main__':

    import argparse
    for i in range(3,12,2):
        parser = argparse.ArgumentParser(description="Mask R-CNN Model Evaluation")
        parser.add_argument('--device', default='cuda:0', help='Device to use for inference')
        parser.add_argument('--data-path', default=r'./WBC_Dataset/active-learning-4-cell1/randomchoose', help='Path to dataset')
        parser.add_argument('--weights', default='./run_result/active2/'+str(i)+'com/best.pth',
                            help='Path to trained model weights')
        parser.add_argument('--num-classes', default=4, type=int, help='Number of classes (excluding background)')

        args = parser.parse_args()
        eval(args)