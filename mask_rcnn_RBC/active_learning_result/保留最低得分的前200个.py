import csv
import os
import shutil

def move_lowest_scoring_images(csv_file, source_folder, destination_folder):
    data = []
    try:
        # 读取 CSV 文件并计算得分
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # 跳过标题行
            for row in reader:
                values_sum = sum(map(float, row[1:]))  # 转换最后三列为浮点数并求和

                data.append((row[0], values_sum))  # 保存文件名和求和值
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # 找到最低得分的前 200 行数据
    lowest_files = sorted(data, key=lambda x: x[1])[:200]#######
    print(lowest_files)
    lowest_files = [str(f[0]) for f in lowest_files]  # 提取文件名列表

    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)

    # 移动图片到目标文件夹
    for filename in lowest_files:
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        try:
            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
                print(f"Moved: {filename}")
            else:
                print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error moving file {filename}: {e}")

# 示例调用
# csv_file = r'D:\work\Active_learning\mask_rcnn\active-learning-result\results1.csv'  # 替换为实际的 CSV 文件路径
# source_folder = r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\improve'  # 替换为存放图片的原始文件夹路径
# destination_folder = r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell0\improve_200'  # 替换为目标文件夹路径
# move_lowest_scoring_images(csv_file, source_folder, destination_folder)
