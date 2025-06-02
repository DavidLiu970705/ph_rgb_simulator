import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from skimage import color

st.set_page_config(page_title="pH 與 RGB 對應模擬器", layout="wide")

# ===================== 資料定義 =====================

data_mode = st.sidebar.radio("選擇資料來源", ("垂直拍攝", "側拍攝"))

# --- 垂直拍攝數據（已平均）
ph_vertical = np.array([1.4, 2.4, 3.4, 4.4, 7, 9, 10, 11, 12])
r_vertical = np.array([140, 132, 126, 112, 124, 121, 120, 110, 126])
g_vertical = np.array([114, 117, 124, 122, 128, 126, 129, 123, 128])
b_vertical = np.array([117, 122, 127, 124, 131, 130, 134, 129, 104])

# --- 側拍攝數據（保留雙點）
ph_side = np.array([1.4, 1.4, 2.4, 2.4, 3.4, 3.4, 4.4, 4.4, 7, 7, 9, 9, 10, 10, 11, 11, 12, 12])
r_side = np.array([111, 110, 78, 73, 46, 52, 30, 32, 50, 59, 29, 39, 30, 26, 44, 42, 35, 47])
g_side = np.array([39, 38, 35, 28, 33, 42, 49, 50, 70, 74, 47, 55, 48, 44, 71, 70, 37, 51])
b_side = np.array([59, 58, 62, 57, 63, 75, 63, 64, 81, 79, 59, 70, 60, 58, 78, 74, 23, 37])

# --- 切換數據來源
if data_mode == "垂直拍攝":
    ph_values, r_values, g_values, b_values = ph_vertical, r_vertical, g_vertical, b_vertical
else:
    ph_values, r_values, g_values, b_values = ph_side, r_side, g_side, b_side

# ===================== 擬合與轉換 =====================

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

def rgb_to_lab(rgb):
    rgb_scaled = np.array(rgb) / 255.0
    lab = color.rgb2lab(rgb_scaled.reshape(1, 1, 3)).reshape(3)
    return lab

def rgb_to_hsv(rgb):
    rgb_scaled = np.array(rgb) / 255.0
    hsv = color.rgb2hsv(rgb_scaled.reshape(1, 1, 3)).reshape(3)
    return hsv

# ===================== 側邊欄 =====================

with st.sidebar:
    st.title("pH 控制面板")
    ph_input = st.slider("選擇 pH 值", 1.0, 13.0, 7.0, step=0.01)
    r, g, b = get_rgb(ph_input)
    st.markdown(f"**RGB:** ({r}, {g}, {b})")
    st.markdown(f"**HEX:** #{r:02x}{g:02x}{b:02x}")
    lab = rgb_to_lab((r, g, b))
    hsv = rgb_to_hsv((r, g, b))
    st.markdown(f"**Lab:** ({lab[0]:.2f}, {lab[1]:.2f}, {lab[2]:.2f})")
    st.markdown(f"**HSV:** ({hsv[0]*360:.1f}°, {hsv[1]*100:.1f}%, {hsv[2]*100:.1f}%)")
    st.markdown(
        f'<div style="width:120px;height:60px;background:#{r:02x}{g:02x}{b:02x};border-radius:10px;border:2px solid #888;margin-bottom:10px"></div>',
        unsafe_allow_html=True
    )

# ===================== 主畫面 =====================

col1, col2 = st.columns(2)

with col1:
    st.subheader("pH 對 RGB 曲線圖")
    x_plot = np.linspace(1, 13, 200)
    fig, ax = plt.subplots()
    ax.plot(x_plot, poly2(x_plot, *params_r), 'r-', label='R')
    ax.plot(x_plot, poly2(x_plot, *params_g), 'g-', label='G')
    ax.plot(x_plot, poly2(x_plot, *params_b), 'b-', label='B')

    # 真實數據點
    ax.scatter(ph_values, r_values, color='r', s=50, alpha=0.6, label="R 點")
    ax.scatter(ph_values, g_values, color='g', s=50, alpha=0.6, label="G 點")
    ax.scatter(ph_values, b_values, color='b', s=50, alpha=0.6, label="B 點")

    # 即時點
    ax.axvline(ph_input, color='gray', linestyle='--')
    ax.plot(ph_input, poly2(ph_input, *params_r), 'ro')
    ax.plot(ph_input, poly2(ph_input, *params_g), 'go')
    ax.plot(ph_input, poly2(ph_input, *params_b), 'bo')
    ax.set_xlabel("pH")
    ax.set_ylabel("RGB 值")
    ax.set_ylim(0, 255)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("RGB 三維曲線圖")
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    r_curve = poly2(x_plot, *params_r)
    g_curve = poly2(x_plot, *params_g)
    b_curve = poly2(x_plot, *params_b)

    # 曲線
    ax3d.plot(r_curve, g_curve, b_curve, label="RGB 曲線", color="black", lw=2)
    ax3d.scatter(*get_rgb(ph_input), s=100, c=[[r/255, g/255, b/255]])

    # 原始數據點
    ax3d.scatter(r_values, g_values, b_values, c=np.array([r_values, g_values, b_values]).T / 255, s=40)

    ax3d.set_xlabel("R")
    ax3d.set_ylabel("G")
    ax3d.set_zlabel("B")

    ax3d.set_xlim(0, 150)
    ax3d.set_ylim(0, 150)
    ax3d.set_zlim(0, 150)
    ax3d.view_init(elev=30, azim=45)
    st.pyplot(fig3d)
    plt.close(fig3d)

# ===================== 漸層預覽 =====================

st.subheader("pH 變化漸層預覽")
gradient_fig, gradient_ax = plt.subplots(figsize=(10, 1))
ph_range = np.linspace(1.0, 13.0, 256)
gradient_colors = [np.array(get_rgb(ph)) / 255.0 for ph in ph_range]
gradient_ax.imshow([gradient_colors], extent=[1, 13, 0, 1])
gradient_ax.set_yticks([])
gradient_ax.set_xlabel("pH")
st.pyplot(gradient_fig)
plt.close(gradient_fig)

# ===================== 匯出 CSV =====================

st.download_button("匯出目前 RGB 成 CSV",
                   data=f"pH,R,G,B\n{ph_input},{r},{g},{b}",
                   file_name="ph_rgb.csv",
                   mime="text/csv")

