import os
import datetime

import torch
from torchvision.ops.misc import FrozenBatchNorm2d

import transforms
from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from my_dataset_coco import CocoDetection
from my_dataset_voc import VOCInstances
from train_utils import train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups





def create_model(num_classes, load_pretrain_weights=True):
    # 如果GPU显存很小，batch_size不能设置很大，建议将norm_layer设置成FrozenBatchNorm2d(默认是nn.BatchNorm2d)
    # FrozenBatchNorm2d的功能与BatchNorm2d类似，但参数无法更新
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    # backbone = resnet50_fpn_backbone(norm_layer=FrozenBatchNorm2d,
    #                                  trainable_layers=3)
    # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth
    backbone = resnet50_fpn_backbone(pretrain_path="resnet50.pth", trainable_layers=3)

    model = MaskRCNN(backbone, num_classes=num_classes)

    if load_pretrain_weights:
        # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        weights_dict = torch.load("./maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
        for k in list(weights_dict.keys()):
            if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                del weights_dict[k]

        print(model.load_state_dict(weights_dict, strict=False))

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # det_results_file = f"det_results{now}.txt"
    # seg_results_file = f"seg_results{now}.txt"
    det_results_file = os.path.join(output_folder, f"det_results{now}.txt")
    seg_results_file = os.path.join(output_folder, f"seg_results{now}.txt")


    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    data_root = args.data_path

    # load train data set
    # coco2017 -> annotations -> instances_train2017.json
    train_dataset = CocoDetection(data_root, "train", data_transform["train"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    # train_dataset = VOCInstances(data_root, year="2012", txt_name="train.txt", transforms=data_transform["train"])
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    # coco2017 -> annotations -> instances_val2017.json
    val_dataset = CocoDetection(data_root, "val", data_transform["val"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    # val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt", transforms=data_transform["val"])
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)

    # create model num_classes equal background + classes
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain)
    model.to(device)
    model.backbone.fpn.set_inference_mode(False)

    train_loss = []
    learning_rate = []
    val_map = []

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_steps,
                                                        gamma=args.lr_gamma)
    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
    best_map = 0.0  # 初始化最佳mAP50
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        det_info, seg_info = utils.evaluate(model, val_data_loader, device=device)

        # write detection into txt
        with open(det_results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        # write seg into txt
        with open(seg_results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(det_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        # torch.save(save_files, "./run_result/7/model_{}.pth".format(epoch))
        # torch.save(save_files, "./"+output_folder+"/last.pth".format(epoch))

        # 保存最佳权重
        current_map50 = seg_info[1]  # seg的mAP50
        if current_map50 > best_map:
            best_map = current_map50
            torch.save(save_files, "./" + output_folder + "/best.pth")
            print(f"Best mAP50 updated to {best_map:.4f} at epoch {epoch}")
        '''修改文件路径'''
    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate,files=output_folder)
    '''修改文件路径'''
    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map,files=output_folder)
import random
import shutil
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
        shutil.move(image_source_path, image_target_path)
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
def copy_improve(csv_file,source_folder,destination_folder):
    import os
    import shutil
    import pandas as pd

    # 读取 CSV 文件，忽略第一行
    df = pd.read_csv(csv_file)

    # 获取文件名列表，忽略第一行
    file_names = df.iloc[:, 0].tolist()

    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)

    # 复制匹配的文件
    for file_name in file_names:
        src_path = os.path.join(source_folder, file_name)
        dst_path = os.path.join(destination_folder, file_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"已复制: {file_name}")
        else:
            print(f"文件未找到: {file_name}")

    print("文件复制完成！")
'''修改文件路径'''
if __name__ == "__main__":
    import argparse
    from active_learning_result.对比实验随机选择增强数据 import *
    for j in range (9,10):
        data_folder = r"./RBC_Dataset/active_learning0" + str(j)
        # 源文件夹
        src_folder = data_folder+r"/train2017_00"
        # 目标文件夹
        dst_folder = data_folder+"/randomchoose/train2017_00"
        dst_folder2 = data_folder + "/randomchoose/train2017"
        shutil.copytree(src_folder, dst_folder)
        shutil.copytree(src_folder, dst_folder2)
        #copy test
        src_folder_test = data_folder + r"/test2017"
        dst_folder_test = data_folder + "/randomchoose/test2017"
        shutil.copytree(src_folder_test, dst_folder_test)
        src_path = data_folder+ r'/annotations/instances_test2017.json'
        dst_path = data_folder+ r"/randomchoose/annotations/instances_test2017.json"
        dst_dir = os.path.dirname(dst_path)

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)  # 递归创建目标文件夹
        shutil.copy(src_path, dst_path)
        #copy val
        src_folder_val = data_folder + r"/val2017"
        dst_folder_val = data_folder + "/randomchoose/val2017"
        shutil.copytree(src_folder_val, dst_folder_val)
        src_path = data_folder+ r'/annotations/instances_val2017.json'
        dst_path = data_folder+ r"/randomchoose/annotations/instances_val2017.json"
        shutil.copy(src_path, dst_path)
        # 设置文件路径
        csv_file = data_folder+"/result/1/results1.csv"  # CSV 文件路径
        source_folder = "./RBC_Dataset/all"  # 图片所在的文件夹
        destination_folder = data_folder+"/randomchoose/improve"  # 目标文件夹
        copy_improve(csv_file,source_folder,destination_folder)


        for i in range (2,12):
            import argparse
            from RBC.get_annotation import create_coco_json, split
            '''
            1..数据划分
            2. 训练
            3. 不确定分析
            4.筛选200个
            5. 从200个中选10个
            6. 训练
            7 不确定分析
            8. 选10个
            9. 训练
            '''

            output_folder = data_folder+"/randomchoose/result/"+str(i)

            source_image_folder = destination_folder
            base_target_folder = data_folder+r"\randomchoose\train2017"
            num_total = 6* (i-1)

            target_folder = prepare_target_folder(base_target_folder)
            selected_files = select_random_files(source_image_folder, num_total)
            print('selected_files:', selected_files)
            copy_selected_files(selected_files, source_image_folder, target_folder)
            print("图片和标签选择及复制完成！")


            # # 数据集路径
            TRAIN_IMAGE_DIR = base_target_folder
            MASK_DIR = r"./RBC_Dataset/COCO_Annotations_png"  # 所有标签的文件夹
            OUTPUT_TRAIN_JSON =  data_folder + r'/randomchoose\annotations/instances_train2017.json'


            # 生成训练集和测试集的 JSON 文件
            create_coco_json(TRAIN_IMAGE_DIR, OUTPUT_TRAIN_JSON, MASK_DIR)


             #2.训练


            '''修改文件路径'''
            parser = argparse.ArgumentParser(
                description=__doc__)
            # 训练设备类型
            parser.add_argument('--device', default='cuda:0', help='device')
            # 训练数据集的根目录
            parser.add_argument('--data-path', default=data_folder+"/randomchoose/", help='dataset')
            # 检测目标类别数(不包含背景)
            parser.add_argument('--num-classes', default=8, type=int, help='num_classes')
            # 文件保存地址
            parser.add_argument('--output-dir', default=output_folder, help='path where to save')
            '''修改文件路径'''
            # 若需要接着上次训练，则指定上次训练保存权重文件地址
            parser.add_argument('--resume', default=False, type=str, help='resume from checkpoint')
            # 指定接着从哪个epoch数开始训练
            parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
            # 训练的总epoch数
            parser.add_argument('--epochs', default=10, type=int, metavar='N',
                                help='number of total epochs to run')
            # 学习率  x
            parser.add_argument('--lr', default=0.004, type=float,
                                help='initial learning rate, 0.02 is the default value for training '
                                     'on 8 gpus and 2 images_per_gpu')
            # SGD的momentum参数
            parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                                help='momentum')
            # SGD的weight_decay参数
            parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                                metavar='W', help='weight decay (default: 1e-4)',
                                dest='weight_decay')
            # 针对torch.optim.lr_scheduler.MultiStepLR的参数
            parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                                help='decrease lr every step-size epochs')
            # 针对torch.optim.lr_scheduler.MultiStepLR的参数
            parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
            # 训练的batch size(如果内存/GPU显存充裕，建议设置更大)
            parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                                help='batch size when training.')
            parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
            parser.add_argument("--pretrain", type=bool, default=True, help="load COCO pretrain weights.")
            # 是否使用混合精度训练(需要GPU支持混合精度)
            parser.add_argument("--amp", default=True, help="Use torch.cuda.amp for mixed precision training")

            args = parser.parse_args()
            # print(args)

            # 检查保存权重文件夹是否存在，不存在则创建
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            main(args)

            ##评估
            from active_learning_result.evaluate import eval
            import argparse

            parser_eval = argparse.ArgumentParser(description="Mask R-CNN Model Evaluation")
            parser_eval.add_argument('--device', default='cuda:0', help='Device to use for inference')
            parser_eval.add_argument('--data-path',
                                default=data_folder+"/randomchoose/",
                                help='Path to dataset')
            parser_eval.add_argument('--weights', default=output_folder+'/best.pth',
                                help='Path to trained model weights')
            parser_eval.add_argument('--num-classes', default=8, type=int, help='Number of classes (excluding background)')

            args_eval = parser_eval.parse_args()
            eval(args_eval)

            #3.不确定分析


            # from active_learning_result.fast_uncertainty_analysis import unce_anal_sta
            #
            # weights_path =  output_folder+r"\\best.pth"
            # results_csv_path = output_folder+r'\\results'+str(i)+'.csv'
            # folder_path = r"./RBC_Dataset"+data_folder+r"\\improve"
            # unce_anal_sta(weights_path,results_csv_path,folder_path)

            #4.选取200个
            # if i == 1:
            #
            #     from active_learning_result.保留最低得分的前200个 import move_lowest_scoring_images
            #
            #     csv_file = r"D:\work\Active_learning\mask_rcnn\\"+ output_folder+r'\results1.csv'  # 替换为实际的 CSV 文件路径
            #     source_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset"+data_folder+"/improve"  # 替换为存放图片的原始文件夹路径
            #     destination_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset"+data_folder+"/improve_200"  # 替换为目标文件夹路径
            #     move_lowest_scoring_images(csv_file, source_folder, destination_folder)






