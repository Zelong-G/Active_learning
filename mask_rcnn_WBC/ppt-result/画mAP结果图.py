import matplotlib.pyplot as plt
import numpy as np

# Data
epochs = list(range(40))
map90 = [
    0.3621, 0.4327, 0.4222, 0.5104, 0.5970, 0.6246, 0.5997, 0.6229, 0.6006, 0.5918,
    0.5996, 0.5876, 0.6105, 0.5960, 0.6086, 0.6479, 0.6528, 0.6436, 0.6324, 0.6409,
    0.6396, 0.6365, 0.6524, 0.6380, 0.6437, 0.6443, 0.6348, 0.6357, 0.6402, 0.6491,
    0.6530, 0.6449, 0.6634, 0.6484, 0.6423, 0.6351, 0.6378, 0.6451, 0.6376, 0.6400
]
map50 = [0.4394, 0.4964, 0.4887, 0.571, 0.6817, 0.711, 0.6949, 0.7223, 0.6757, 0.6718,
 0.6823, 0.6654, 0.6915, 0.6777, 0.6941, 0.7293, 0.7394, 0.7266, 0.7169, 0.7248,
 0.7246, 0.7227, 0.7403, 0.7236, 0.7289, 0.7299, 0.7184, 0.7203, 0.727, 0.7398,
 0.7398, 0.7304, 0.7518, 0.7342, 0.7285, 0.7206, 0.7236, 0.7323, 0.7242, 0.7263]


# Colors
colors_map90 = plt.cm.plasma(np.linspace(0, 1, len(epochs)))
colors_map50 = plt.cm.viridis(np.linspace(0, 1, len(epochs)))

# Plot
plt.figure(figsize=(10, 8))

# Plot map90
for i in range(len(epochs) - 1):
    plt.plot(epochs[i:i + 2], map90[i:i + 2], color=colors_map90[i], linewidth=2)
plt.fill_between(epochs, np.array(map90) - 0.01, np.array(map90) + 0.01, color='orange', alpha=0.2)

# Plot map50
for i in range(len(epochs) - 1):
    plt.plot(epochs[i:i + 2], map50[i:i + 2], color=colors_map50[i], linewidth=2)
plt.fill_between(epochs, np.array(map50) - 0.01, np.array(map50) + 0.01, color='green', alpha=0.2)

# Highlights
max_idx_map90 = np.argmax(map90)
max_idx_map50 = np.argmax(map50)
plt.scatter(epochs[max_idx_map90], map90[max_idx_map90], color='red', label=f"Max map90: {map90[max_idx_map90]:.3f}",s=150, zorder=8)
plt.scatter(epochs[max_idx_map50], map50[max_idx_map50], color='blue', label=f"Max map50: {map50[max_idx_map50]:.3f}",s=150, zorder=8)

# Titles and labels
plt.title("mAP50 and mAP90 Over Epochs", fontsize=25, fontweight='bold')
plt.xlabel("Epochs", fontsize=24, fontweight='bold')
plt.ylabel("mAP", fontsize=24, fontweight='bold')

# Styling
plt.gca().set_facecolor('#f5f5f5')
plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='gray')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=37, loc='lower right')

# Adjust layout and show
plt.tight_layout()
plt.show()

def mAP_multi():
    import matplotlib.pyplot as plt

    # 数据
    e1_map50 = [0.6396, 0.8285, 0.8516, 0.8648, 0.9259, 0.9274, 0.9174, 0.9146, 0.9208, 0.9164]
    e2_map50 = [0.1162, 0.1678, 0.2745, 0.2696, 0.305, 0.3179, 0.364, 0.3721, 0.4268, 0.4428, 0.4124]
    e3_map50 = [0.1495, 0.1675, 0.1924, 0.2281, 0.2452, 0.2675, 0.3023, 0.3361, 0.3539, 0.3629, 0.3739, 0.4695, 0.3975,
                0.4798, 0.3933, 0.4348, 0.4817, 0.4899, 0.4375, 0.4488, 0.4452, 0.4392, 0.5226, 0.4362, 0.4839, 0.5397,
                0.4619, 0.5174, 0.4398, 0.4535, 0.4875, 0.4898, 0.4537, 0.4377, 0.4796, 0.4665, 0.4759, 0.4417, 0.4867,
                0.4535]
    e4_map50 = [0.4502, 0.4975, 0.4915, 0.571, 0.6827, 0.711, 0.6949, 0.7223, 0.6757, 0.6718, 0.6823, 0.6654, 0.6915,
                0.6777, 0.6941, 0.7293, 0.7394, 0.7266, 0.7169, 0.7248, 0.7246, 0.7227, 0.7403, 0.7236, 0.7289, 0.7299,
                0.7184, 0.7203, 0.727, 0.7398, 0.7398, 0.7304, 0.7518, 0.7342, 0.7285, 0.7206, 0.7236, 0.7323, 0.7242,
                0.7263]
    e5_map50 = [0.0875, 0.1259, 0.1976, 0.1993, 0.2442, 0.283, 0.314, 0.34, 0.3836, 0.4254, 0.4316, 0.5409, 0.5285,
                0.4892, 0.5108, 0.4954, 0.5551, 0.5386, 0.6115, 0.5753, 0.5535, 0.5547, 0.4809, 0.5785, 0.6045, 0.574,
                0.541, 0.5705, 0.5647, 0.591, 0.6012, 0.5571, 0.5755, 0.5593, 0.5608, 0.547, 0.5491, 0.5805, 0.5913,
                0.5337]
    e6_map50 = [0.4249, 0.4142, 0.4743, 0.4898, 0.6225, 0.7116, 0.7403, 0.7383, 0.7661, 0.7751, 0.6519, 0.7707, 0.7605,
                0.7665, 0.7684, 0.7618, 0.7729, 0.7694, 0.7951, 0.7656, 0.7812, 0.7654, 0.7986, 0.7652, 0.777, 0.7748,
                0.7924, 0.7734, 0.7967, 0.7723, 0.7906, 0.793, 0.7931, 0.7689, 0.785, 0.7933, 0.8069, 0.7933, 0.8007,
                0.7803]

    # 绘图
    plt.figure(figsize=(6, 6))

    plt.plot(e1_map50, label='Exp-1', marker='o', linestyle='-', linewidth=2)
    plt.plot(e2_map50, label='Exp-2', marker='s', linestyle='--', linewidth=2)
    plt.plot(e3_map50, label='Exp-3', marker='^', linestyle='-.', linewidth=2)
    plt.plot(e4_map50, label='Exp-4', marker='*', linestyle=':', linewidth=2)
    plt.plot(e5_map50, label='Exp-5', marker='x', linestyle='-', linewidth=2)
    plt.plot(e6_map50, label='Exp-6', marker='D', linestyle='--', linewidth=2)
    # 为每条线添加末尾标签
    plt.text(len(e1_map50) - 1, e1_map50[-1], 'Exp-1', fontsize=12, ha='left', va='center')
    plt.text(len(e2_map50) - 1, e2_map50[-1], 'Exp-2', fontsize=12, ha='left', va='center')
    plt.text(len(e3_map50) - 1, e3_map50[-1], 'Exp-3', fontsize=12, ha='left', va='center')
    plt.text(len(e4_map50) - 1, e4_map50[-1], 'Exp-4', fontsize=12, ha='left', va='center')
    plt.text(len(e5_map50) - 1, e5_map50[-1], 'Exp-5', fontsize=12, ha='left', va='center')
    plt.text(len(e6_map50) - 1, e6_map50[-1], 'Exp-6', fontsize=12, ha='left', va='center')
    # 添加标题和标签
    plt.title('MAP@50 for Different Experiments', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('MAP@50', fontsize=14)

    # 显示图例
    plt.legend(fontsize=14)

    # 添加网格和显示
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
mAP_multi()