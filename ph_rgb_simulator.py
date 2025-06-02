import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

st.set_page_config(page_title="pH 與 RGB 對應模擬器", layout="wide")

# 側拍與垂拍原始資料
ph_side = np.array([1.4, 1.4, 2.4, 2.4, 3.4, 3.4, 4.4, 4.4, 7, 7, 9, 9, 10, 10, 11, 11, 12, 12])
r_side = np.array([111, 110, 78, 73, 46, 52, 30, 32, 50, 59, 29, 39, 30, 26, 44, 42, 35, 47])
g_side = np.array([39, 38, 35, 28, 33, 42, 49, 50, 70, 74, 47, 55, 48, 44, 71, 70, 37, 51])
b_side = np.array([59, 58, 62, 57, 63, 75, 63, 64, 81, 79, 59, 70, 60, 58, 78, 74, 23, 37])

ph_top = np.array([1.4, 2.4, 3.4, 4.4, 7, 9, 10, 11, 12])
r_top = np.array([140, 132, 126, 112, 124, 121, 120, 110, 126])
g_top = np.array([114, 117, 124, 122, 128, 126, 129, 123, 128])
b_top = np.array([117, 122, 127, 124, 131, 130, 134, 129, 104])

# 選擇資料來源
st.sidebar.title("pH 控制面板")
data_view = st.sidebar.radio("選擇資料來源", ["側拍", "垂拍"])

if data_view == "側拍":
    ph_values = ph_side
    r_values = r_side
    g_values = g_side
    b_values = b_side
else:
    ph_values = ph_top
    r_values = r_top
    g_values = g_top
    b_values = b_top

# 平方擬合
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

# pH 輸入
ph_input = st.sidebar.slider("選擇 pH 值", 1.0, 13.0, 7.0, step=0.01)
r, g, b = get_rgb(ph_input)
st.sidebar.markdown(f"**RGB:** ({r}, {g}, {b})")
st.sidebar.markdown(f"**HEX:** #{r:02x}{g:02x}{b:02x}")
st.sidebar.markdown(
    f'<div style="width:120px;height:60px;background:#{r:02x}{g:02x}{b:02x};border-radius:10px;border:2px solid #888;margin-bottom:10px"></div>',
    unsafe_allow_html=True
)

# 主區域
col1, col2 = st.columns(2)

# 曲線圖
with col1:
    st.subheader("pH 對 RGB 曲線圖")
    x_plot = np.linspace(1, 13, 200)
    fig, ax = plt.subplots()
    ax.plot(x_plot, poly2(x_plot, *params_r), 'r-', label='R')
    ax.plot(x_plot, poly2(x_plot, *params_g), 'g-', label='G')
    ax.plot(x_plot, poly2(x_plot, *params_b), 'b-', label='B')
    ax.axvline(ph_input, color='gray', linestyle='--')
    ax.plot(ph_input, r, 'ro')
    ax.plot(ph_input, g, 'go')
    ax.plot(ph_input, b, 'bo')
    ax.set_xlabel("pH")
    ax.set_ylabel("RGB 值")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# 3D 圖
with col2:
    st.subheader("RGB 三維曲線圖")
    fig_3d = plt.figure()
    ax3d = fig_3d.add_subplot(111, projection='3d')
    r_curve = poly2(x_plot, *params_r)
    g_curve = poly2(x_plot, *params_g)
    b_curve = poly2(x_plot, *params_b)
    ax3d.plot(r_curve, g_curve, b_curve, label="RGB 曲線", color="black")
    ax3d.scatter(r, g, b, s=100, c=[[r/255, g/255, b/255]])
    ax3d.set_xlabel("R")
    ax3d.set_ylabel("G")
    ax3d.set_zlabel("B")
    ax3d.set_xlim(0, 255)
    ax3d.set_ylim(0, 255)
    ax3d.set_zlim(0, 255)
    plt.tight_layout()
    st.pyplot(fig_3d)
    plt.close(fig_3d)

# 漸層圖
st.subheader("RGB 漸層預覽")
gradient_colors = [get_rgb(p) for p in np.linspace(1, 13, 100)]
gradient_html = "".join([
    f'<div style="flex:1;height:40px;background:rgb({r},{g},{b})"></div>' 
    for r, g, b in gradient_colors
])
st.markdown(
    f'<div style="display:flex;border:1px solid #444;border-radius:5px;overflow:hidden">{gradient_html}</div>',
    unsafe_allow_html=True
)

# 匯出按鈕
st.download_button("匯出目前 RGB 成 CSV", 
                   data=f"pH,R,G,B\\n{ph_input},{r},{g},{b}", 
                   file_name="ph_rgb.csv",
                   mime="text/csv")
