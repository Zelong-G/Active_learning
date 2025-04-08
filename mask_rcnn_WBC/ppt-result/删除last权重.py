import os

def delete_last_files(directory):
    """遍历目录并删除名为 'last' 的文件"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "last.pth":
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除 {file_path} 失败: {e}")

if __name__ == "__main__":
    folder_path = r"D:\work\Active_learning\mask_rcnn\run_result\active3"
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        delete_last_files(folder_path)
    else:
        print("指定的路径不存在或不是一个文件夹。")
