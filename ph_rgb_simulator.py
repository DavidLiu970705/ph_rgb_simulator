import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from skimage import color

st.set_page_config(page_title="pH 與 RGB 對應模擬器", layout="wide")

# 原始數據（垂拍）
ph_vertical = np.array([1.4, 1.4, 2.4, 2.4, 3.4, 3.4, 4.4, 4.4, 7, 7, 9, 9, 10, 10, 11, 11, 12, 12])
r_vertical = np.array([111, 110, 78, 73, 46, 52, 30, 32, 50, 59, 29, 39, 30, 26, 44, 42, 35, 47])
g_vertical = np.array([39, 38, 35, 28, 33, 42, 49, 50, 70, 74, 47, 55, 48, 44, 71, 70, 37, 51])
b_vertical = np.array([59, 58, 62, 57, 63, 75, 63, 64, 81, 79, 59, 70, 60, 58, 78, 74, 23, 37])

# 側拍數據（平均後）
ph_side = np.array([1.4, 2.4, 3.4, 4.4, 7, 9, 10, 11, 12])
r_side = np.array([140, 132, 126, 112, 124, 121, 120, 110, 126])
g_side = np.array([114, 117, 124, 122, 128, 126, 129, 123, 128])
b_side = np.array([117, 122, 127, 124, 131, 130, 134, 129, 104])

# 曲線擬合函式
def poly2(x, a, b, c):
    return a * x**2 + b * x + c

def fit_and_predict(ph_vals, r_vals, g_vals, b_vals):
    pr, _ = curve_fit(poly2, ph_vals, r_vals)
    pg, _ = curve_fit(poly2, ph_vals, g_vals)
    pb, _ = curve_fit(poly2, ph_vals, b_vals)
    return pr, pg, pb

# 選擇資料集
with st.sidebar:
    st.title("pH 控制面板")
    mode = st.radio("選擇資料集", ["垂直拍攝", "側拍攝"])
    ph_input = st.slider("pH 值", 1.0, 13.0, 7.0, step=0.01)

if mode == "垂直拍攝":
    ph_vals, r_vals, g_vals, b_vals = ph_vertical, r_vertical, g_vertical, b_vertical
    unit_range = 140
else:
    ph_vals, r_vals, g_vals, b_vals = ph_side, r_side, g_side, b_side
    unit_range = 260

# 擬合
params_r, params_g, params_b = fit_and_predict(ph_vals, r_vals, g_vals, b_vals)

def get_rgb(ph):
    r = int(np.clip(poly2(ph, *params_r), 0, 255))
    g = int(np.clip(poly2(ph, *params_g), 0, 255))
    b = int(np.clip(poly2(ph, *params_b), 0, 255))
    return r, g, b

r, g, b = get_rgb(ph_input)
rgb_normalized = np.array([[[r/255, g/255, b/255]]])
hsv = color.rgb2hsv(rgb_normalized)[0][0]
lab = color.rgb2lab(rgb_normalized)[0][0]

# 顯示數值與色塊
with st.sidebar:
    st.markdown(f"**RGB：** ({r}, {g}, {b})")
    st.markdown(f"**HSV：** ({hsv[0]:.2f}, {hsv[1]:.2f}, {hsv[2]:.2f})")
    st.markdown(f"**Lab：** ({lab[0]:.2f}, {lab[1]:.2f}, {lab[2]:.2f})")
    st.markdown(f"**HEX：** #{r:02x}{g:02x}{b:02x}")
    st.markdown(
        f'<div style="width:120px;height:60px;background:#{r:02x}{g:02x}{b:02x};border-radius:10px;border:2px solid #888;margin-bottom:10px"></div>',
        unsafe_allow_html=True
    )

# 漸層條
st.subheader("pH 漸層條")
gradient = np.zeros((50, 300, 3), dtype=np.uint8)
for i, pH in enumerate(np.linspace(1, 13, 300)):
    rr, gg, bb = get_rgb(pH)
    gradient[:, i, :] = [rr, gg, bb]
st.image(gradient, caption="pH 1 ~ 13 色彩漸層", use_container_width=True)

# --- 圖表 ---
x_plot = np.linspace(1, 13, 200)
r_curve = poly2(x_plot, *params_r)
g_curve = poly2(x_plot, *params_g)
b_curve = poly2(x_plot, *params_b)

col1, col2 = st.columns(2)

with col1:
    st.subheader("pH 對 RGB 曲線圖")
    fig, ax = plt.subplots()
    ax.plot(x_plot, r_curve, 'r-', label='R 曲線')
    ax.plot(x_plot, g_curve, 'g-', label='G 曲線')
    ax.plot(x_plot, b_curve, 'b-', label='B 曲線')
    ax.scatter(ph_vals, r_vals, color='red', s=25, label='R 實際值')
    ax.scatter(ph_vals, g_vals, color='green', s=25, label='G 實際值')
    ax.scatter(ph_vals, b_vals, color='blue', s=25, label='B 實際值')
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
    st.subheader("RGB 三維曲線圖")
    fig_3d = plt.figure()
    ax3d = fig_3d.add_subplot(111, projection='3d')
    ax3d.plot(r_curve, g_curve, b_curve, color='black', label="擬合曲線")
    ax3d.scatter(r_vals, g_vals, b_vals, c=np.array([r_vals, g_vals, b_vals]).T / 255, s=50, label="實際數據點")
    ax3d.scatter(r, g, b, s=100, c=[[r/255, g/255, b/255]], label="當前 pH 點", edgecolor="black")
    ax3d.set_xlabel("R")
    ax3d.set_ylabel("G")
    ax3d.set_zlabel("B")
    ax3d.set_xlim(0, unit_range)
    ax3d.set_ylim(0, unit_range)
    ax3d.set_zlim(0, unit_range)
    ax3d.legend()
    plt.tight_layout()
    st.pyplot(fig_3d)
    plt.close(fig_3d)

# --- 匯出 CSV ---
csv_data = f"pH,R,G,B\n{ph_input},{r},{g},{b}"
st.download_button("匯出目前 RGB 成 CSV", data=csv_data, file_name="ph_rgb.csv", mime="text/csv")
