# 读取文本文件
file_path = r"D:\work\Active_learning\mask_rcnn\完整数据-去噪音-40epoch-无dropout-seg_results20250110-125447.txt"  # 替换为你的文件路径

# 存储第二列数据的列表
second_column = []

# 打开文件并逐行读取
with open(file_path, "r") as file:
    for line in file:
        if line.startswith("epoch:"):  # 确保只处理以 "epoch:" 开头的行
            parts = line.split()       # 按空格分割行内容
            if len(parts) > 1:         # 确保第二列存在
                second_column.append(float(parts[2]))  # 将第二列数据转换为浮点数并添加到列表

# 打印结果
print(second_column)
