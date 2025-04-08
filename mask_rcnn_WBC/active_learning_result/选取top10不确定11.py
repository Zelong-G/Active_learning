import csv
import os
import shutil

def process_and_copy_files(csv_file, improve_folder, train2017_folder, improvelabel_folder, trainlabel_folder, base_folder):
    """
    从 CSV 文件中读取数据，找到最后三列和最小的 10 个文件，并将其对应的图片和标签复制到目标文件夹。

    参数：
        csv_file (str): CSV 文件路径。
        improve_folder (str): 原始图片所在文件夹路径。
        train2017_folder (str): 目标图片文件夹路径。
        improvelabel_folder (str): 原始标签文件夹路径。
        trainlabel_folder (str): 目标标签文件夹路径。
    """
    # 读取 CSV 文件并计算每行最后三个值的和
    data = []
    try:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # 跳过标题行
            for row in reader:
                values_sum = sum(map(float, row[1:]))  # 转换最后三列为浮点数并求和
                data.append((row[0], values_sum))  # 保存文件名和求和值
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # 找到求和值最低的 10 行数据
    lowest_files = sorted(data, key=lambda x: x[1])[:10]
    print(lowest_files)
    lowest_files = [str(f[0]) for f in lowest_files]


    step_number = 1

    # 找到下一个可用的 step 文件夹
    while True:
        step_folder = os.path.join(base_folder, f"step{step_number}")
        if not os.path.exists(step_folder):
            os.makedirs(step_folder)  # 创建文件夹
            print(f"Created folder: {step_folder}")
            break
        step_number += 1
    # step_image_folder = os.path.join(step_folder, "image")
    # os.makedirs(step_image_folder)  # 创建文件夹
    # step_label_folder = os.path.join(step_folder, "label")
    # os.makedirs(step_label_folder)  # 创建文件夹

    step_image_folder = step_folder
    # 复制图片到目标 step-image 文件夹

    for file_name in lowest_files:
        source_image = os.path.join(improve_folder, file_name)
        target_image = os.path.join(step_image_folder, file_name)
        if os.path.exists(source_image):
            shutil.move(source_image, target_image)
            print(f"Copied {source_image} to {target_image}")
        else:
            print(f"Image not found: {source_image}")


    # # 复制图片到目标 step-label 文件夹
    # for file_name in lowest_files:
    #     source_label = os.path.join(improvelabel_folder, file_name)
    #     target_image = os.path.join(step_label_folder, file_name)
    #     if os.path.exists(source_label):
    #         shutil.copy(source_label, target_image)
    #         print(f"Copied {source_label} to {target_image}")
    #     else:
    #         print(f"Image not found: {source_label}")

# 示例调用
csv_file = r'D:\work\Active_learning\mask_rcnn\active-learning-result\results6.csv'
improve_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell\improve_200"
train2017_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell\train2017"
improvelabel_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell\improvelabel"
trainlabel_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell\train2017label"
base_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell"
process_and_copy_files(csv_file, improve_folder, train2017_folder, improvelabel_folder, trainlabel_folder,base_folder)


