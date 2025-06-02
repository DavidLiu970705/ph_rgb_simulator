import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from skimage import color
import colorsys

st.set_page_config(page_title="pH 與 RGB 對應模擬器", layout="wide")

# --- 原始數據 ---
ph_values_side = np.array([1.4, 2.4, 3.4, 4.4, 7, 9, 10, 11, 12])
r_side = np.array([140, 132, 126, 112, 124, 121, 120, 110, 126])
g_side = np.array([114, 117, 124, 122, 128, 126, 129, 123, 128])
b_side = np.array([117, 122, 127, 124, 131, 130, 134, 129, 104])

ph_values_top = np.array([1.4, 1.4, 2.4, 2.4, 3.4, 3.4, 4.4, 4.4, 7, 7, 9, 9, 10, 10, 11, 11, 12, 12])
r_top = np.array([111, 110, 78, 73, 46, 52, 30, 32, 50, 59, 29, 39, 30, 26, 44, 42, 35, 47])
g_top = np.array([39, 38, 35, 28, 33, 42, 49, 50, 70, 74, 47, 55, 48, 44, 71, 70, 37, 51])
b_top = np.array([59, 58, 62, 57, 63, 75, 63, 64, 81, 79, 59, 70, 60, 58, 78, 74, 23, 37])

def poly2(x, a, b, c):
    return a * x**2 + b * x + c

def fit_polynomials(ph_values, r_values, g_values, b_values):
    pr, _ = curve_fit(poly2, ph_values, r_values)
    pg, _ = curve_fit(poly2, ph_values, g_values)
    pb, _ = curve_fit(poly2, ph_values, b_values)
    return pr, pg, pb

def get_rgb(ph, pr, pg, pb):
    r = int(np.clip(poly2(ph, *pr), 0, 255))
    g = int(np.clip(poly2(ph, *pg), 0, 255))
    b = int(np.clip(poly2(ph, *pb), 0, 255))
    return r, g, b

# --- 側邊欄 ---
with st.sidebar:
    st.title("pH 控制面板")
    dataset = st.radio("選擇資料集", ["垂直拍攝", "側拍"])
    ph_input = st.slider("pH 值", 1.0, 13.0, 7.0, step=0.01)

    if dataset == "垂直拍攝":
        ph_values = ph_values_top
        r_values = r_top
        g_values = g_top
        b_values = b_top
    else:
        ph_values = ph_values_side
        r_values = r_side
        g_values = g_side
        b_values = b_side

    pr, pg, pb = fit_polynomials(ph_values, r_values, g_values, b_values)
    r, g, b = get_rgb(ph_input, pr, pg, pb)

    st.markdown(f"**RGB:** ({r}, {g}, {b})")
    st.markdown(f"**HEX:** #{r:02x}{g:02x}{b:02x}")

    # Lab
    rgb_norm = np.array([[[r / 255, g / 255, b / 255]]])
    lab = color.rgb2lab(rgb_norm)[0][0]
    st.markdown(f"**Lab:** ({lab[0]:.2f}, {lab[1]:.2f}, {lab[2]:.2f})")

    # HSV
    hsv = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    st.markdown(f"**HSV:** ({hsv[0]*360:.1f}°, {hsv[1]*100:.1f}%, {hsv[2]*100:.1f}%)")

    st.markdown(
        f'<div style="width:120px;height:60px;background:#{r:02x}{g:02x}{b:02x};border-radius:10px;border:2px solid #888;margin-bottom:10px"></div>',
        unsafe_allow_html=True
    )

# --- 主區域 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("pH 對 RGB 曲線圖")
    x_plot = np.linspace(1, 13, 300)
    fig, ax = plt.subplots()
    ax.plot(x_plot, poly2(x_plot, *pr), 'r-', label='R')
    ax.plot(x_plot, poly2(x_plot, *pg), 'g-', label='G')
    ax.plot(x_plot, poly2(x_plot, *pb), 'b-', label='B')
    ax.axvline(ph_input, color='gray', linestyle='--')
    ax.plot(ph_input, r, 'ro')
    ax.plot(ph_input, g, 'go')
    ax.plot(ph_input, b, 'bo')
    ax.set_xlabel("pH")
    ax.set_ylabel("RGB 值")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # 顏色漸層條
    st.markdown("#### 色彩漸層預覽")
    gradient_fig, gradient_ax = plt.subplots(figsize=(6, 1))
    gradient_ax.axis('off')
    gradient = np.linspace(1, 13, 300)
    gradient_rgb = np.array([get_rgb(ph, pr, pg, pb) for ph in gradient], dtype=np.uint8)
    gradient_img = np.reshape(gradient_rgb, (1, 300, 3))
    gradient_ax.imshow(gradient_img, aspect='auto')
    st.pyplot(gradient_fig)

with col2:
    st.subheader("RGB 三維曲線圖")
    fig_3d = plt.figure()
    ax3d = fig_3d.add_subplot(111, projection='3d')
    r_curve = poly2(x_plot, *pr)
    g_curve = poly2(x_plot, *pg)
    b_curve = poly2(x_plot, *pb)

    # 畫曲線與當前點
    ax3d.plot(r_curve, g_curve, b_curve, color='black', label='RGB Curve')
    ax3d.scatter(r, g, b, s=100, c=[[r / 255, g / 255, b / 255]], label='Current Color')

    # 若為側拍也加上所有點
    if dataset == "垂直拍攝":
        ax3d.scatter(r_values, g_values, b_values, c=np.array([r_values, g_values, b_values]).T / 255.0, s=40, label="Data Points")

    ax3d.set_xlabel("R")
    ax3d.set_ylabel("G")
    ax3d.set_zlabel("B")

    # 根據資料調整顯示範圍
    max_rgb = max(r_values.max(), g_values.max(), b_values.max(), 150)
    ax3d.set_xlim(0, max_rgb)
    ax3d.set_ylim(0, max_rgb)
    ax3d.set_zlim(0, max_rgb)
    ax3d.legend()
    plt.tight_layout()
    st.pyplot(fig_3d)

# --- 匯出 CSV ---
st.download_button(
    "匯出目前 RGB 成 CSV",
    data=f"pH,R,G,B\n{ph_input},{r},{g},{b}",
    file_name="ph_rgb.csv",
    mime="text/csv"
)
