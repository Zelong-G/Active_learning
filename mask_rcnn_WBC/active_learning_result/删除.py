import os

# 设置目标文件夹路径
target_dir = r"D:\work\Active_learning\mask_rcnn\run_result\active3"  # 修改为你的目标文件夹路径

# 遍历文件夹
for i in range(1, 12):  # 处理 1-11 和 1com-11com 目录
    folders = [str(i), f"{i}com"]
    for folder in folders:
        folder_path = os.path.join(target_dir, folder)
        last_pth_path = os.path.join(folder_path, "last.pth")

        if os.path.exists(last_pth_path):  # 检查文件是否存在
            try:
                os.remove(last_pth_path)  # 删除文件
                print(f"Deleted: {last_pth_path}")
            except Exception as e:
                print(f"Failed to delete {last_pth_path}: {e}")
        else:
            print(f"File not found: {last_pth_path}")
