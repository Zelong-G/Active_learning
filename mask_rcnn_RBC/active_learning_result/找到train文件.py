import os
import shutil

def get_jpg_files(folder_path):
    """获取文件夹中所有的 .jpg 文件名"""
    return set(f for f in os.listdir(folder_path) if f.endswith('.jpg'))

def find_unique_jpgs(main_folder, folders_to_exclude):
    """找到主文件夹中不在其他文件夹中的 .jpg 文件"""
    # 获取主文件夹中的 .jpg 文件
    main_jpgs = get_jpg_files(main_folder)

    # 获取其他文件夹中的 .jpg 文件
    exclude_jpgs = set()
    for folder in folders_to_exclude:
        exclude_jpgs.update(get_jpg_files(folder))

    # 找到主文件夹中不在排除列表中的文件
    unique_jpgs = main_jpgs - exclude_jpgs
    return unique_jpgs

def copy_unique_jpgs(main_folder, folders_to_exclude, destination_folder):
    """将唯一的 .jpg 文件复制到目标文件夹"""
    unique_jpgs = find_unique_jpgs(main_folder, folders_to_exclude)

    # 创建目标文件夹（如果不存在）
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 复制文件
    for jpg in unique_jpgs:
        src_path = os.path.join(main_folder, jpg)
        dest_path = os.path.join(destination_folder, jpg)
        shutil.copy(src_path, dest_path)

# 示例用法
if __name__ == "__main__":
    main_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell\4-cell-data"  # 替换为主文件夹路径
    folders_to_exclude = [
        r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell\improve",  # 替换为第一个排除文件夹路径
        r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell\test2017",  # 替换为第二个排除文件夹路径
        r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell\val2017"   # 替换为第三个排除文件夹路径
    ]
    destination_folder = r"D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning-4-cell\train100"  # 替换为目标文件夹路径

    copy_unique_jpgs(main_folder, folders_to_exclude, destination_folder)
    print(f"唯一的 .jpg 文件已复制到: {destination_folder}")
