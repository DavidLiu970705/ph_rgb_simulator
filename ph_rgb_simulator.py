import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

st.set_page_config(page_title="pH 與 RGB 對應模擬器", layout="wide")

# ===== 數據定義區 =====
# 垂直拍攝
ph_vertical = np.array([1.4, 1.4, 2.4, 2.4, 3.4, 3.4, 4.4, 4.4, 7, 7, 9, 9, 10, 10, 11, 11, 12, 12])
r_vertical = np.array([111, 110, 78, 73, 46, 52, 30, 32, 50, 59, 29, 39, 30, 26, 44, 42, 35, 47])
g_vertical = np.array([39, 38, 35, 28, 33, 42, 49, 50, 70, 74, 47, 55, 48, 44, 71, 70, 37, 51])
b_vertical = np.array([59, 58, 62, 57, 63, 75, 63, 64, 81, 79, 59, 70, 60, 58, 78, 74, 23, 37])

# 側拍攝
ph_side = np.array([1.4, 2.4, 3.4, 4.4, 7, 9, 10, 11, 12])
r_side = np.array([140, 132, 126, 112, 124, 121, 120, 110, 126])
g_side = np.array([114, 117, 124, 122, 128, 126, 129, 123, 128])
b_side = np.array([117, 122, 127, 124, 131, 130, 134, 129, 104])

# ===== 擬合函數 =====
def poly2(x, a, b, c):
    return a * x**2 + b * x + c

def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d

def fit_model(ph, values, model):
    if model == "Quadratic":
        return curve_fit(poly2, ph, values)[0], poly2
    elif model == "Sigmoid":
        return curve_fit(sigmoid, ph, values, maxfev=10000)[0], sigmoid

# ===== 側邊選單 =====
with st.sidebar:
    st.title("pH 控制面板")
    dataset_choice = st.selectbox("選擇數據集", ["垂直拍攝", "側拍攝"])
    model_choice = st.selectbox("擬合模型", ["Quadratic", "Sigmoid"])
    ph_input = st.slider("選擇 pH 值", 1.0, 13.0, 7.0, step=0.01)

# ===== 擷取數據 =====
if dataset_choice == "垂直拍攝":
    ph_values, r_values, g_values, b_values = ph_vertical, r_vertical, g_vertical, b_vertical
else:
    ph_values, r_values, g_values, b_values = ph_side, r_side, g_side, b_side

# ===== 模型擬合 =====
params_r, func_r = fit_model(ph_values, r_values, model_choice)
params_g, func_g = fit_model(ph_values, g_values, model_choice)
params_b, func_b = fit_model(ph_values, b_values, model_choice)

def get_rgb(ph):
    r = int(np.clip(func_r(ph, *params_r), 0, 255))
    g = int(np.clip(func_g(ph, *params_g), 0, 255))
    b = int(np.clip(func_b(ph, *params_b), 0, 255))
    return r, g, b

r, g, b = get_rgb(ph_input)

# ===== 顯示顏色與數值 =====
st.markdown(f"**RGB:** ({r}, {g}, {b})")
st.markdown(f"**HEX:** #{r:02x}{g:02x}{b:02x}")
st.markdown(
    f'<div style="width:120px;height:60px;background:#{r:02x}{g:02x}{b:02x};border-radius:10px;border:2px solid #888;margin-bottom:10px"></div>',
    unsafe_allow_html=True
)

# ===== 色彩漸層條 =====
st.subheader("pH 色彩漸層條")
gradient = np.linspace(1.0, 13.0, 200)
grad_colors = [get_rgb(p) for p in gradient]
grad_img = np.array([[grad_colors[i] for i in range(len(gradient))]], dtype=np.uint8)
fig_grad, ax_grad = plt.subplots(figsize=(8, 1))
ax_grad.imshow(grad_img, aspect="auto")
ax_grad.set_xticks([])
ax_grad.set_yticks([])
st.pyplot(fig_grad)

# ===== 二維 RGB 曲線圖 =====
st.subheader("pH 對 RGB 曲線圖")
x_plot = np.linspace(1.0, 13.0, 300)
fig2d, ax2d = plt.subplots()
ax2d.plot(x_plot, func_r(x_plot, *params_r), 'r-', label='R')
ax2d.plot(x_plot, func_g(x_plot, *params_g), 'g-', label='G')
ax2d.plot(x_plot, func_b(x_plot, *params_b), 'b-', label='B')
ax2d.axvline(ph_input, color='gray', linestyle='--')
ax2d.plot(ph_input, func_r(ph_input, *params_r), 'ro')
ax2d.plot(ph_input, func_g(ph_input, *params_g), 'go')
ax2d.plot(ph_input, func_b(ph_input, *params_b), 'bo')
ax2d.set_xlabel("pH")
ax2d.set_ylabel("RGB 值")
ax2d.legend()
plt.tight_layout()
st.pyplot(fig2d)

# ===== 三維 RGB 空間圖 =====
st.subheader("RGB 三維曲線圖")
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
r_curve = func_r(x_plot, *params_r)
g_curve = func_g(x_plot, *params_g)
b_curve = func_b(x_plot, *params_b)
ax3d.plot(r_curve, g_curve, b_curve, label="RGB Curve", color="black")
ax3d.scatter(*get_rgb(ph_input), s=100, c=[[r/255, g/255, b/255]])
ax3d.set_xlabel("R")
ax3d.set_ylabel("G")
ax3d.set_zlabel("B")
ax3d.set_xlim(0, 255)
ax3d.set_ylim(0, 255)
ax3d.set_zlim(0, 255)
plt.tight_layout()
st.pyplot(fig3d)

# ===== 匯出 CSV 按鈕 =====
st.download_button("匯出目前 RGB 成 CSV", 
                   data=f"pH,R,G,B\n{ph_input},{r},{g},{b}", 
                   file_name="ph_rgb.csv",
                   mime="text/csv")

