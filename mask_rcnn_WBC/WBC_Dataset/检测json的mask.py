import json

# JSON 文件路径
json_file_path = r'D:\work\Active_learning\mask_rcnn\WBC_Dataset\active-learning\annotations\instances_improve.json'


# 读取 JSON 文件
with open(json_file_path, 'r') as file:
    data = json.load(file)

# 获取 annotations 部分
annotations = data.get("annotations", [])

# 初始化检查结果
invalid_records = []

# 遍历 annotations 检查 segmentation
for record in annotations:
    segmentation = record.get('segmentation', [])
    if len(segmentation) != 1:  # 如果 segmentation 的长度不为 1
        invalid_records.append(record['id'])

# 输出检测结果
if invalid_records:
    print(f"以下记录的 segmentation 不符合要求（ID 列表）：{invalid_records}")
else:
    print("所有 segmentation 字段都符合要求。")

