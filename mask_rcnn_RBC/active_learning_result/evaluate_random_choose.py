import os
import torch
from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from my_dataset_coco import CocoDetection
from train_utils import train_eval_utils as utils
import transforms
import os
import torch
from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from my_dataset_coco import CocoDetection
from train_utils import train_eval_utils as utils
import torch.utils.data as data
from pycocotools.coco import COCO
from PIL import Image
def create_model(num_classes):
    # 创建模型
    backbone = resnet50_fpn_backbone(pretrain_path="../resnet50.pth", trainable_layers=3)
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device.type))

    # 数据预处理
    data_transform = {
        "test": transforms.Compose([transforms.ToTensor()])
    }

    # 加载测试数据集
    data_root = args.data_path
    val_dataset = CocoDetection(data_root, "test", data_transform["test"])

    # 创建测试数据加载器
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,  # 根据你的硬件调整
        collate_fn=val_dataset.collate_fn
    )

    # 创建模型并加载权重
    model = create_model(num_classes=args.num_classes + 1)  # num_classes 包括背景
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.backbone.fpn.set_inference_mode(False)
    model.to(device)
    model.eval()

    # 评估模型
    det_info, seg_info = utils.evaluate(model, val_data_loader, device=device)

    # 获取权重所在目录
    weights_dir = os.path.dirname(args.weights)
    results_file = os.path.join(weights_dir, "evaluation_results.txt")

    # 将结果写入文件
    with open(results_file, "w") as f:
        f.write("Detection Results:\n")
        f.write(f"AP: {det_info[0]:.4f}, AP50: {det_info[1]:.4f}, AP75: {det_info[2]:.4f}\n")
        f.write("Segmentation Results:\n")
        f.write(f"AP: {seg_info[0]:.4f}, AP50: {seg_info[1]:.4f}, AP75: {seg_info[2]:.4f}\n")

    print(f"Evaluation results saved to {results_file}")

if __name__ == "__main__":
    import argparse
    for i in range(2,12):
        parser = argparse.ArgumentParser(description="Mask R-CNN Model Evaluation")
        parser.add_argument('--device', default='cuda:0', help='Device to use for inference')
        parser.add_argument('--data-path', default=r'D:\work\Active_learning\mask_rcnn\WBC_Dataset/active-learning-4-cell0', help='Path to dataset')
        weight = 'D:\work\Active_learning\mask_rcnn/run_result/active/'+str(i)+'com/best.pth'
        parser.add_argument('--weights', default=weight, help='Path to trained model weights')
        parser.add_argument('--num-classes', default=4, type=int, help='Number of classes (excluding background)')

        args = parser.parse_args()
        main(args)



