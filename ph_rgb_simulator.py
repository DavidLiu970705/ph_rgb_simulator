import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

st.set_page_config(page_title="pH 與 RGB 對應模擬器", layout="wide")

# 側拍與垂拍資料
side_data = {
    "ph": np.array([1.4, 2.4, 3.4, 4.4, 7, 9, 10, 11, 12]),
    "r":  np.array([111, 78, 46, 30, 50, 29, 30, 44, 35]),
    "g":  np.array([39, 35, 33, 49, 70, 47, 48, 71, 37]),
    "b":  np.array([59, 62, 63, 63, 81, 59, 60, 78, 23])
}

top_data = {
    "ph": np.array([1.4, 2.4, 3.4, 4.4, 7, 9, 10, 11, 12]),
    "r":  np.array([110, 73, 52, 32, 59, 39, 26, 42, 47]),
    "g":  np.array([38, 28, 42, 50, 74, 55, 44, 70, 51]),
    "b":  np.array([58, 57, 75, 64, 79, 70, 58, 74, 37])
}

# 側邊欄選單
with st.sidebar:
    st.title("pH 控制面板")
    data_type = st.radio("選擇拍攝角度", ["垂拍", "側拍"])
    ph_input = st.slider("選擇 pH 值", 1.0, 13.0, 7.0, step=0.01)

# 選資料
data = top_data if data_type == "垂拍" else side_data
ph_values = data["ph"]
r_values = data["r"]
g_values = data["g"]
b_values = data["b"]

# 曲線擬合
def poly2(x, a, b, c):
    return a * x**2 + b * x + c

params_r, _ = curve_fit(poly2, ph_values, r_values)
params_g, _ = curve_fit(poly2, ph_values, g_values)
params_b, _ = curve_fit(poly2, ph_values, b_values)

def get_rgb(ph):
    r = int(np.clip(poly2(ph, *params_r), 0, 255))
    g = int(np.clip(poly2(ph, *params_g), 0, 255))
    b = int(np.clip(poly2(ph, *params_b), 0, 255))
    return r, g, b

# 計算目前 RGB
r, g, b = get_rgb(ph_input)

# 顯示目前色彩
with st.sidebar:
    st.markdown(f"**RGB:** ({r}, {g}, {b})")
    st.markdown(f"**HEX:** #{r:02x}{g:02x}{b:02x}")
    st.markdown(
        f'<div style="width:120px;height:60px;background:#{r:02x}{g:02x}{b:02x};border-radius:10px;border:2px solid #888;margin-bottom:10px"></div>',
        unsafe_allow_html=True
    )

# 主內容
col1, col2 = st.columns(2)

with col1:
    st.subheader("RGB 曲線圖")
    x_plot = np.linspace(1, 13, 200)
    fig, ax = plt.subplots()
    ax.plot(x_plot, poly2(x_plot, *params_r), 'r-', label='R')
    ax.plot(x_plot, poly2(x_plot, *params_g), 'g-', label='G')
    ax.plot(x_plot, poly2(x_plot, *params_b), 'b-', label='B')
    ax.scatter(ph_values, r_values, color='r', s=30, label='R 點')
    ax.scatter(ph_values, g_values, color='g', s=30, label='G 點')
    ax.scatter(ph_values, b_values, color='b', s=30, label='B 點')
    ax.axvline(ph_input, color='gray', linestyle='--')
    ax.plot(ph_input, poly2(ph_input, *params_r), 'ro')
    ax.plot(ph_input, poly2(ph_input, *params_g), 'go')
    ax.plot(ph_input, poly2(ph_input, *params_b), 'bo')
    ax.set_xlabel("pH")
    ax.set_ylabel("RGB 值")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("RGB 三維分布圖")
    fig_3d = plt.figure()
    ax3d = fig_3d.add_subplot(111, projection='3d')
    r_curve = poly2(x_plot, *params_r)
    g_curve = poly2(x_plot, *params_g)
    b_curve = poly2(x_plot, *params_b)
    ax3d.plot(r_curve, g_curve, b_curve, color="black", label="RGB 曲線")
    ax3d.scatter(r_values, g_values, b_values, c=np.stack([r_values, g_values, b_values], axis=1)/255.0, s=40, label='資料點')
    ax3d.scatter(r, g, b, s=100, c=[[r/255, g/255, b/255]], edgecolors='black', label='目前點')
    ax3d.set_xlabel("R")
    ax3d.set_ylabel("G")
    ax3d.set_zlabel("B")
    ax3d.set_xlim(0, 255)
    ax3d.set_ylim(0, 255)
    ax3d.set_zlim(0, 255)
    ax3d.set_box_aspect([2, 2, 2])  # 調整 XYZ 比例，讓點不密集
    plt.tight_layout()
    st.pyplot(fig_3d)
    plt.close(fig_3d)

# 顏色漸層圖
st.subheader("pH 色彩漸層預覽")
gradient = np.linspace(1, 13, 200)
gradient_colors = np.array([get_rgb(ph) for ph in gradient]) / 255.0

fig_grad, ax_grad = plt.subplots(figsize=(8, 1.2))
ax_grad.imshow([gradient_colors], extent=[1, 13, 0, 1], aspect='auto')
ax_grad.set_yticks([])
ax_grad.set_xlabel("pH")
plt.tight_layout()
st.pyplot(fig_grad)
plt.close(fig_grad)

# 匯出 CSV
st.download_button("匯出目前 RGB 成 CSV", 
                   data=f"pH,R,G,B\n{ph_input},{r},{g},{b}", 
                   file_name="ph_rgb.csv",
                   mime="text/csv")
