import numpy as np
import scipy.stats


coco_categories = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# 映射类别 ID 到类别名称
category_names = {i + 1: name for i, name in enumerate(coco_categories)}


def calculate_uncertainty(results, config=None, category_names=category_names, rare_list=None, rare_thresh=0.5):
    """
    计算检测、分割、分类的不确定性，以及稀有实例数量。
    Args:
        results (list): 每个元素为检测模型的输出，包括 masks, rois, class_ids, probs。
        config (object): 配置对象，包含是否计算稀有实例的开关等。
        category_names (list): COCO 数据集的类别名称。
        rare_list (list): 稀有类别的列表。
        rare_thresh (float): 稀有实例的概率阈值。
    Returns:
        tuple: 各类不确定性的均值、每个实例的不确定性列表、稀有实例计数。
    """
    detection_uncertainties = []
    segmentation_uncertainties = []
    classification_uncertainties = []
    instance_details = []

    masks, rois, class_ids = extract_data_from_results(results)

    # 找到包含最多检测框的结果作为 anchor
    anchor_id = find_anchor_id(rois)
    anchor = rois.pop(anchor_id)
    anchor_masks = masks.pop(anchor_id)
    anchor_classes = class_ids.pop(anchor_id)
    '''从基准的anchor中一个一个提取实例0，然后可以在其他结果中提取与这个实例0对应的实例，存储起来。
        就能得到属于不同结果中的同个实例。
        other_rois意思是存储不同结果中同个实例的数据'''
    for anchor_idx, anchor_roi in enumerate(anchor):
        other_rois, other_masks, other_classes = align_with_anchor(
            anchor_roi, rois, masks, class_ids, anchor_masks, anchor_classes, anchor_idx
        )
        '''得到一个实例在不同结果的数据后，就可以进行该实例的不确定分析了'''
        detection_uncertainties.append(measure_detection_uncertainty(other_rois))
        segmentation_uncertainties.append(measure_segmentation_uncertainty(other_masks))
        classification_uncertainties.append(measure_classification_uncertainty(other_classes))
        instance_name = category_names[str(anchor_classes[anchor_idx])]
        instance_details.append(f"Instance {anchor_idx} ({instance_name})")

    rare_cells_count = count_rare_instances(results, config, rare_list, rare_thresh)

    # 打印计算结果
    # Print calculated results
    # print("Segmentation Uncertainty (Instance -> Value):")
    # for detail, value in zip(instance_details, segmentation_uncertainties):
    #     print(f"{detail}: {value}")
    #
    # print("Mean Segmentation Uncertainty:", compute_mean(segmentation_uncertainties))
    #
    # print("Detection Uncertainty (Instance -> Value):")
    # for detail, value in zip(instance_details, detection_uncertainties):
    #     print(f"{detail}: {value}")
    #
    # print("Mean Detection Uncertainty:", compute_mean(detection_uncertainties))
    #
    # print("Classification Uncertainty (Instance -> Value):")
    # for detail, value in zip(instance_details, classification_uncertainties):
    #     print(f"{detail}: {value}")
    #
    # print("Mean Classification Uncertainty:", compute_mean(classification_uncertainties))

    return (
        compute_mean(segmentation_uncertainties), segmentation_uncertainties,
        compute_mean(detection_uncertainties), detection_uncertainties,
        compute_mean(classification_uncertainties), classification_uncertainties,

    )


def extract_data_from_results(results):
    """提取 masks, rois, class_ids 数据"""
    masks = [r['masks'] for r in results]
    rois = [r['rois'] for r in results]
    class_ids = [r['class_ids'] for r in results]
    return masks, rois, class_ids


def find_anchor_id(rois):
    """根据检测框数量找到 anchor 的索引"""
    return np.argmax([r.shape[0] for r in rois])


def align_with_anchor(anchor_roi, rois, masks, class_ids, anchor_masks, anchor_classes, anchor_idx):
    """
    对齐其他模型的结果与 anchor。
    """
    other_rois, other_masks, other_classes = [], [], []
    '''遍历除了anchor所在的其他每一个结果，如果我们有十个结果，这里的rois有九个结果'''
    for i in range(len(rois)):
        '''#对 anchor_roi 和当前结果的所有 rois 计算 IOU。
        # 找到 IOU 最大的检测框索引 roi_id，即认为它是与 anchor_roi 最匹配的检测框。'''
        roi_id = find_corresponding_roi(anchor_roi, rois[i], compute_iou)
        other_rois.append(rois[i][roi_id])
        other_masks.append(masks[i][roi_id])
        other_classes.append(class_ids[i][roi_id])

    # 将 anchor 的自身结果加入
    other_rois.append(anchor_roi)
    other_masks.append(anchor_masks[ anchor_idx])
    other_classes.append(anchor_classes[anchor_idx])

    return other_rois, other_masks, other_classes


def measure_detection_uncertainty(rois):
    """计算检测框的不确定性（基于 IOU）。"""
    '''计算anchor和其他结果对应的实例的iou，然后取均值'''
    return compute_mean([compute_iou(rois[i], rois[0]) for i in range(1, len(rois))])


def measure_segmentation_uncertainty(masks):
    """计算分割的不确定性（基于 Dice 系数）。"""
    '''计算anchor和其他结果对应的实例的dice，然后取均值'''
    return compute_mean([compute_dice(masks[i], masks[0]) for i in range(1, len(masks))])


def measure_classification_uncertainty(classes):
    """计算分类的不确定性（基于类别分布）。"""
    '''取众数mode'''
    mode_count = scipy.stats.mode(classes).count[0]
    return mode_count / len(classes)


def count_rare_instances(results, config, rare_list, rare_thresh):
    """统计稀有实例的数量。"""
    if not getattr(config, 'RARE_INSTANCES', False) or not rare_list:
        return 0

    rare_cells_count = 0
    for r in results:
        probs = r['probs']
        rare_probs = probs.T[rare_list].T
        rare_cells_count += sum((rare_probs > rare_thresh).sum())

    return rare_cells_count


def find_corresponding_roi(anchor_roi, rois, metric):
    """找到与 anchor 对齐的检测框（基于给定的度量标准）。"""
    ious = [metric(anchor_roi, roi) for roi in rois]
    return np.argmax(ious)


def compute_mean(values):
    """计算列表的均值，避免空列表报错。"""
    return np.mean(values) if values else 0


def compute_iou(box1, box2):
    """计算两个检测框的 IOU（占位函数）。"""
    intersection = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * \
                   max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def compute_dice(mask1, mask2):
    """计算两个分割掩码的 Dice 系数（占位函数）。"""
    intersection = np.sum(mask1 * mask2)
    return 2 * intersection / (np.sum(mask1) + np.sum(mask2))
