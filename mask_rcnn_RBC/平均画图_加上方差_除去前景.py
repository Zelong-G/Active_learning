import matplotlib.pyplot as plt
import numpy as np

# 生成 X 轴
x_values = [20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80]  # 过滤 X 轴数据

# 生成 Y 轴范围，与原数据保持一致
y_min, y_max = 0.18, 0.42  # 根据上传的图片估计合适的 Y 轴范围
x_min, x_max = 20, 80
# 创建图像
plt.figure(figsize=(12, 8))

# 设置 X 轴和 Y 轴标签
plt.xlabel("Size of Training Set", size=18, fontweight='bold')
plt.ylabel("Segmentation AP50", size=18, fontweight='bold')
plt.title("RBC Segmentation AP50 Comparison", size=18, fontweight='bold')

# 仅显示网格，不绘制任何曲线
plt.xticks(np.linspace(x_min, x_max, num=11))  # 设置 X 轴刻度
plt.yticks(np.linspace(y_min, y_max, num=6))  # 设定 Y 轴刻度，与原图一致
plt.ylim(y_min, y_max)  # 设置 Y 轴范围
plt.xlim(x_min, x_max)  # 设置 Y 轴范围
plt.grid(True)

# 保存图片，格式可以是 'png', 'pdf', 'svg' 等
plt.savefig("RBC_segmentation_ap50_background.png", dpi=800, bbox_inches='tight')

plt.show()
