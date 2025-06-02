import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

# 資料（色碼、RGB、pH）
data = [
    ("#8c7275", [140, 114, 117], 1.4),
    ("#84757a", [132, 117, 122], 2.4),
    ("#7e7c7f", [126, 124, 127], 3.4),
    ("#707a7c", [112, 122, 124], 4.4),
    ("#7c8083", [124, 128, 131], 7),
    ("#797e82", [121, 126, 130], 9),
    ("#788186", [120, 129, 134], 10),
    ("#6e7b81", [110, 123, 129], 11),
    ("#7e8068", [126, 128, 104], 12),
]

# 轉換資料為 numpy 陣列
rgb = np.array([d[1] for d in data])
ph = np.array([d[2] for d in data])

# 定義擬合函數：多項式函數
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

# 對每個通道進行二次曲線擬合
params_r, _ = curve_fit(quadratic, ph, rgb[:, 0])
params_g, _ = curve_fit(quadratic, ph, rgb[:, 1])
params_b, _ = curve_fit(quadratic, ph, rgb[:, 2])

# 生成擬合曲線資料點
ph_range = np.linspace(min(ph), max(ph), 300)
r_fit = quadratic(ph_range, *params_r)
g_fit = quadratic(ph_range, *params_g)
b_fit = quadratic(ph_range, *params_b)

# 建立3D圖形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 原始資料點
ax.scatter(rgb[:, 0], rgb[:, 1], rgb[:, 2], c=ph, cmap='viridis', s=60, label='Data')

# 擬合曲線
ax.plot(r_fit, g_fit, b_fit, color='red', linewidth=2, label='Fitted Curve')

# 座標軸標籤與單位調整
ax.set_xlabel('Red Channel (×1)', fontsize=12)
ax.set_ylabel('Green Channel (×1)', fontsize=12)
ax.set_zlabel('Blue Channel (×1)', fontsize=12)

# 將顯示範圍放大一點
ax.set_xlim([100, 150])
ax.set_ylim([100, 150])
ax.set_zlim([90, 140])

ax.set_title('3D RGB vs pH Curve (Quadratic Fit)', fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
