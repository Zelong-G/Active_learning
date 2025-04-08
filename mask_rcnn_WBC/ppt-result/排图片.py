import os
from PIL import Image

# 设置图片文件夹路径
image_folder = r"D:\work\Active_learning\mask_rcnn\ppt-result\mask"  # 替换为你的文件夹路径
output_path = r"D:\work\Active_learning\mask_rcnn\ppt-result\mask/collage.png"  # 输出拼接图片的路径

# 设置布局行数和列数
rows = 3
cols = 6

# 获取文件夹中的图片文件
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

# 检查图片数量是否满足需求
if len(image_files) < rows * cols:
    raise ValueError(f"图片数量不足，文件夹中只有 {len(image_files)} 张图片，但需要 {rows * cols} 张图片")

# 打开所有图片
images = [Image.open(img) for img in image_files[:rows * cols]]

# 获取图片尺寸（假定所有图片尺寸一致）
width, height = images[0].size

# 创建一个新图像用于拼接
collage_width = cols * width
collage_height = rows * height
collage = Image.new("RGB", (collage_width, collage_height))

# 将图片逐一贴到拼接图像上
for idx, img in enumerate(images):
    row = idx // cols
    col = idx % cols
    x_offset = col * width
    y_offset = row * height
    collage.paste(img, (x_offset, y_offset))

# 保存拼接图像
os.makedirs(os.path.dirname(output_path), exist_ok=True)
collage.save(output_path)
print(f"拼接完成，已保存到 {output_path}")
