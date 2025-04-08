import os

# 替换成你自己的目标文件夹路径
target_folder = r'D:\work\Active_learning'

# 遍历目标文件夹及其所有子目录
for root, dirs, files in os.walk(target_folder):
    for file in files:
        if file == 'best.pth':
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"已删除: {file_path}")
            except Exception as e:
                print(f"删除失败: {file_path}，错误信息: {e}")
