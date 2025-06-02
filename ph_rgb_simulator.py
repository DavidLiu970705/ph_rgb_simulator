# ph_rgb_simulator.py - 完整修正版

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from skimage import color

st.set_page_config(page_title="pH 與 RGB 對應模擬器", layout="wide")

# ==== 垂直拍攝資料（平均後） ====
ph_vertical = np.array([1.4, 2.4, 3.4, 4.4, 7.0, 9.0, 10.0, 11.0, 12.0])
r_vertical = np.array([140, 132, 126, 112, 124, 121, 120, 110, 126])
g_vertical = np.array([114, 117, 124, 122, 128, 126, 129, 123, 128])
b_vertical = np.array([117, 122, 127, 124, 131, 130, 134, 129, 104])

# ==== 側拍資料 ====
ph_side = np.array([1.4, 1.4, 2.4, 2.4, 3.4, 3.4, 4.4, 4.4, 7, 7, 9, 9, 10, 10, 11, 11, 12, 12])
r_side = np.array([111, 110, 78, 73, 46, 52, 30, 32, 50, 59, 29, 39, 30, 26, 44, 42, 35, 47])
g_side = np.array([39, 38, 35, 28, 33, 42, 49, 50, 70, 74, 47, 55, 48, 44, 71, 70, 37, 51])
b_side = np.array([59, 58, 62, 57, 63, 75, 63, 64, 81, 79, 59, 70, 60, 58, 78, 74, 23, 37])

# ==== 多組資料處理 ====
def average_duplicates(ph, r, g, b):
    unique_ph = np.unique(ph)
    avg_r, avg_g, avg_b = [], [], []
    for p in unique_ph:
        mask = ph == p
        avg_r.append(np.mean(r[mask]))
        avg_g.append(np.mean(g[mask]))
        avg_b.append(np.mean(b[mask]))
    return unique_ph, np.array(avg_r), np.array(avg_g), np.array(avg_b)

ph_side_avg, r_side_avg, g_side_avg, b_side_avg = average_duplicates(ph_side, r_side, g_side, b_side)

# ==== 二次多項式擬合 ====
def poly2(x, a, b, c):
    return a * x**2 + b * x + c

def fit_data(ph, values):
    return curve_fit(poly2, ph, values)[0]

# ==== 選擇資料來源 ====
with st.sidebar:
    st.title("pH 控制面板")
    source = st.radio("選擇數據來源", ["側拍", "垂直拍攝"])
    if source == "側拍":
        ph_data, r_data, g_data, b_data = ph_side_avg, r_side_avg, g_side_avg, b_side_avg
        scale = 1.5
    else:
        ph_data, r_data, g_data, b_data = ph_vertical, r_vertical, g_vertical, b_vertical
        scale = 1.0

    # 曲線參數
    params_r = fit_data(ph_data, r_data)
    params_g = fit_data(ph_data, g_data)
    params_b = fit_data(ph_data, b_data)

    # 選擇 pH
    ph_input = st.slider("選擇 pH 值", 1.0, 13.0, 7.0, step=0.01)

# ==== 計算 RGB ====
def get_rgb(ph):
    r = int(np.clip(poly2(ph, *params_r), 0, 255))
    g = int(np.clip(poly2(ph, *params_g), 0, 255))
    b = int(np.clip(poly2(ph, *params_b), 0, 255))
    return r, g, b

r, g, b = get_rgb(ph_input)

# ==== 顏色資訊顯示 ====
with st.sidebar:
    st.markdown(f"**RGB:** ({r}, {g}, {b})")
    st.markdown(f"**HEX:** #{r:02x}{g:02x}{b:02x}")

    # HSV & Lab
    rgb_arr = np.array([[[r/255, g/255, b/255]]])
    hsv = color.rgb2hsv(rgb_arr)[0][0]
    lab = color.rgb2lab(rgb_arr)[0][0]
    st.markdown(f"**HSV:** ({hsv[0]:.2f}, {hsv[1]:.2f}, {hsv[2]:.2f})")
    st.markdown(f"**Lab:** ({lab[0]:.2f}, {lab[1]:.2f}, {lab[2]:.2f})")
    st.markdown(
        f'<div style="width:120px;height:60px;background:#{r:02x}{g:02x}{b:02x};border-radius:10px;border:2px solid #888;margin-bottom:10px"></div>',
        unsafe_allow_html=True
    )

# ==== 色彩漸層 Preview ====
gradient = np.linspace(1, 13, 256)
gradient_rgb = np.array([get_rgb(p) for p in gradient]) / 255
fig_grad, ax_grad = plt.subplots(figsize=(6, 0.8))
ax_grad.imshow([gradient_rgb], aspect="auto")
ax_grad.set_xticks([])
ax_grad.set_yticks([])
ax_grad.set_title("pH 色彩漸層")
st.pyplot(fig_grad, use_container_width=True)
plt.close(fig_grad)

# ==== 主視覺 ===
col1, col2 = st.columns(2)

# ==== 左圖：RGB 曲線圖 ====
with col1:
    st.subheader("pH 對 RGB 曲線圖")
    x_plot = np.linspace(1, 13, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_plot, poly2(x_plot, *params_r), 'r-', label='R')
    ax.plot(x_plot, poly2(x_plot, *params_g), 'g-', label='G')
    ax.plot(x_plot, poly2(x_plot, *params_b), 'b-', label='B')
    ax.scatter(ph_data, r_data, color='r', s=20)
    ax.scatter(ph_data, g_data, color='g', s=20)
    ax.scatter(ph_data, b_data, color='b', s=20)
    ax.axvline(ph_input, color='gray', linestyle='--')
    ax.set_xlabel("pH")
    ax.set_ylabel("RGB 值")
    ax.set_xlim(1, 13)
    ax.set_ylim(0, 255)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ==== 右圖：RGB 三維圖 ====
with col2:
    st.subheader("RGB 三維曲線圖")
    fig_3d = plt.figure(figsize=(6, 6))
    ax3d = fig_3d.add_subplot(111, projection='3d')
    r_curve = poly2(x_plot, *params_r)
    g_curve = poly2(x_plot, *params_g)
    b_curve = poly2(x_plot, *params_b)
    ax3d.plot(r_curve, g_curve, b_curve, label="RGB 曲線", color="black")
    ax3d.scatter(r_data, g_data, b_data, c=np.array([r_data, g_data, b_data]).T/255.0, s=40, label="實測")
    ax3d.scatter(*get_rgb(ph_input), s=120, c=[[r/255, g/255, b/255]], marker='o', label="當前 pH")
    ax3d.set_xlabel("R")
    ax3d.set_ylabel("G")
    ax3d.set_zlabel("B")
    ax3d.set_xlim(0, 180 * scale)
    ax3d.set_ylim(0, 180 * scale)
    ax3d.set_zlim(0, 180 * scale)
    ax3d.legend()
    st.pyplot(fig_3d, use_container_width=True)
    plt.close(fig_3d)

# ==== 匯出 CSV ====
st.download_button("匯出目前 RGB 成 CSV", 
                   data=f"pH,R,G,B\n{ph_input},{r},{g},{b}", 
                   file_name="ph_rgb.csv",
                   mime="text/csv")
