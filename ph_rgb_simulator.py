import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from skimage import color
from matplotlib.colors import rgb_to_hsv

st.set_page_config(page_title="pH 與 RGB 對應模擬器", layout="wide")

# --- 資料集定義 ---
vertical_ph = np.array([1.4, 1.4, 2.4, 2.4, 3.4, 3.4, 4.4, 4.4, 7, 7, 9, 9, 10, 10, 11, 11, 12, 12])
vertical_r = np.array([111, 110, 78, 73, 46, 52, 30, 32, 50, 59, 29, 39, 30, 26, 44, 42, 35, 47])
vertical_g = np.array([39, 38, 35, 28, 33, 42, 49, 50, 70, 74, 47, 55, 48, 44, 71, 70, 37, 51])
vertical_b = np.array([59, 58, 62, 57, 63, 75, 63, 64, 81, 79, 59, 70, 60, 58, 78, 74, 23, 37])

side_ph = np.array([1.4, 2.4, 3.4, 4.4, 7, 9, 10, 11, 12])
side_r = np.array([140, 132, 126, 112, 124, 121, 120, 110, 126])
side_g = np.array([114, 117, 124, 122, 128, 126, 129, 123, 128])
side_b = np.array([117, 122, 127, 124, 131, 130, 134, 129, 104])

# --- 模型擬合 ---
def poly2(x, a, b, c):
    return a * x**2 + b * x + c

# --- 使用者介面 ---
with st.sidebar:
    st.title("pH 控制面板")
    dataset_choice = st.radio("選擇數據來源", ["垂直拍攝", "側拍"])
    ph_input = st.slider("選擇 pH 值", 1.0, 13.0, 7.0, step=0.01)

    if dataset_choice == "垂直拍攝":
        ph_values, r_values, g_values, b_values = vertical_ph, vertical_r, vertical_g, vertical_b
    else:
        ph_values, r_values, g_values, b_values = side_ph, side_r, side_g, side_b

    params_r, _ = curve_fit(poly2, ph_values, r_values)
    params_g, _ = curve_fit(poly2, ph_values, g_values)
    params_b, _ = curve_fit(poly2, ph_values, b_values)

    def get_rgb(ph):
        r = int(np.clip(poly2(ph, *params_r), 0, 255))
        g = int(np.clip(poly2(ph, *params_g), 0, 255))
        b = int(np.clip(poly2(ph, *params_b), 0, 255))
        return r, g, b

    r, g, b = get_rgb(ph_input)
    st.markdown(f"**RGB:** ({r}, {g}, {b})")
    st.markdown(f"**HEX:** #{r:02x}{g:02x}{b:02x}")
    st.markdown(
        f'<div style="width:120px;height:60px;background:#{r:02x}{g:02x}{b:02x};border-radius:10px;border:2px solid #888;margin-bottom:10px"></div>',
        unsafe_allow_html=True
    )

# --- 主內容 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("pH 對 RGB 曲線圖")
    x_plot = np.linspace(1, 13, 300)
    fig, ax = plt.subplots()
    ax.plot(x_plot, poly2(x_plot, *params_r), 'r-', label='R')
    ax.plot(x_plot, poly2(x_plot, *params_g), 'g-', label='G')
    ax.plot(x_plot, poly2(x_plot, *params_b), 'b-', label='B')
    ax.scatter(ph_values, r_values, c='r')
    ax.scatter(ph_values, g_values, c='g')
    ax.scatter(ph_values, b_values, c='b')
    ax.axvline(ph_input, color='gray', linestyle='--')
    ax.set_xlabel("pH")
    ax.set_ylabel("RGB 值")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("RGB 三維曲線圖")
    fig_3d = plt.figure()
    ax3d = fig_3d.add_subplot(111, projection='3d')
    r_curve = poly2(x_plot, *params_r)
    g_curve = poly2(x_plot, *params_g)
    b_curve = poly2(x_plot, *params_b)
    ax3d.plot(r_curve, g_curve, b_curve, label="RGB Curve", color="black")
    ax3d.scatter(*get_rgb(ph_input), s=100, c=[[r/255, g/255, b/255]])
    ax3d.scatter(r_values, g_values, b_values, color='gray', alpha=0.6)
    ax3d.set_xlabel("R")
    ax3d.set_ylabel("G")
    ax3d.set_zlabel("B")
    ax3d.set_xlim(0, 150)
    ax3d.set_ylim(0, 150)
    ax3d.set_zlim(0, 150)
    plt.tight_layout()
    st.pyplot(fig_3d)
    plt.close(fig_3d)

# --- 色彩漸層 ---
st.subheader("pH 漸層色彩預覽")
preview = np.zeros((30, 300, 3), dtype=np.uint8)
for i, ph in enumerate(np.linspace(1, 13, 300)):
    rgb = get_rgb(ph)
    preview[:, i] = rgb
st.image(preview, caption="從 pH 1 到 13 的色彩變化", use_column_width=True)

# --- RGB → HSV / Lab 轉換 ---
st.subheader("色彩空間轉換")
rgb_normalized = np.array([[[r/255, g/255, b/255]]])
hsv = rgb_to_hsv(rgb_normalized)[0][0]
lab = color.rgb2lab(rgb_normalized)[0][0]
st.markdown(f"**HSV:** ({hsv[0]:.2f}, {hsv[1]:.2f}, {hsv[2]:.2f})")
st.markdown(f"**Lab:** (L={lab[0]:.2f}, a={lab[1]:.2f}, b={lab[2]:.2f})")

# --- 匯出 CSV ---
st.download_button("匯出目前 RGB 成 CSV", 
                   data=f"pH,R,G,B\n{ph_input},{r},{g},{b}", 
                   file_name="ph_rgb.csv",
                   mime="text/csv")
