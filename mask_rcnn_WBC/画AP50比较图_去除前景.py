import matplotlib.pyplot as plt
import numpy as np

# 生成 X 轴
x_values = [100,120,140,160,180,200]  # 过滤 X 轴数据

# 生成 Y 轴范围，与原数据保持一致
y_min, y_max = 0.5, 0.8  # 根据上传的图片估计合适的 Y 轴范围
x_min, x_max = 100, 200
# 创建图像
plt.figure(figsize=(12, 8))

# 设置 X 轴和 Y 轴标签
plt.xlabel("Size of Training Set", size=18, fontweight='bold')
plt.ylabel("Segmentation AP50 (Mean ± Std/2)", size=18, fontweight='bold')
plt.title("WBC Segmentation AP50 Comparison (Mean ± Std/2 Dev Over 5 Experiments)", size=18, fontweight='bold')

# 仅显示网格，不绘制任何曲线
plt.xticks(np.linspace(x_min, x_max, num=6))  # 设置 X 轴刻度
plt.yticks(np.linspace(y_min, y_max, num=5))  # 设定 Y 轴刻度，与原图一致
plt.ylim(y_min, y_max)  # 设置 Y 轴范围
plt.xlim(x_min, x_max)  # 设置 Y 轴范围
plt.grid(True)

# 保存图片，格式可以是 'png', 'pdf', 'svg' 等
plt.savefig("WBC_segmentation_ap50_background.png", dpi=800, bbox_inches='tight')

plt.show()
