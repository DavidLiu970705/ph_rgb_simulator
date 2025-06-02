import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

st.set_page_config(page_title="pH 與 RGB 對應模擬器", layout="wide")

# 原始數據
ph_values = np.array([1.4, 1.4, 2.4, 2.4, 3.4, 3.4, 4.4, 4.4, 7, 7, 9, 9, 10, 10, 11, 11, 12, 12])
r_values = np.array([111, 110, 78, 73, 46, 52, 30, 32, 50, 59, 29, 39, 30, 26, 44, 42, 35, 47])
g_values = np.array([39, 38, 35, 28, 33, 42, 49, 50, 70, 74, 47, 55, 48, 44, 71, 70, 37, 51])
b_values = np.array([59, 58, 62, 57, 63, 75, 63, 64, 81, 79, 59, 70, 60, 58, 78, 74, 23, 37])

# 二次曲線擬合
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

# --- 側邊欄 ---
with st.sidebar:
    st.title("pH 控制面板")
    ph_input = st.slider("選擇 pH 值", 1.0, 13.0, 7.0, step=0.01)
    r, g, b = get_rgb(ph_input)
    st.markdown(f"**RGB:** ({r}, {g}, {b})")
    st.markdown(f"**HEX:** #{r:02x}{g:02x}{b:02x}")
    st.markdown(
        f'<div style="width:120px;height:60px;background:#{r:02x}{g:02x}{b:02x};border-radius:10px;border:2px solid #888;margin-bottom:10px"></div>',
        unsafe_allow_html=True
    )

# --- 主區域 ---
col1, col2 = st.columns(2)

# --- 左側曲線圖 ---
with col1:
    st.subheader("pH 對 RGB 曲線圖")
    x_plot = np.linspace(1, 13, 200)

    fig, ax = plt.subplots()
    ax.plot(x_plot, poly2(x_plot, *params_r), 'r-', label='R')
    ax.plot(x_plot, poly2(x_plot, *params_g), 'g-', label='G')
    ax.plot(x_plot, poly2(x_plot, *params_b), 'b-', label='B')
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

# --- 右側 RGB 3D 圖與漸層 ---
with col2:
    st.subheader("RGB 三維曲線圖")
    view_option = st.radio("選擇視角", ["側視圖", "俯視圖"], horizontal=True)

    fig_3d = plt.figure()
    ax3d = fig_3d.add_subplot(111, projection='3d')
    r_curve = poly2(x_plot, *params_r)
    g_curve = poly2(x_plot, *params_g)
    b_curve = poly2(x_plot, *params_b)

    ax3d.plot(r_curve, g_curve, b_curve, label="RGB Curve", color="black")
    ax3d.scatter(*get_rgb(ph_input), s=100, c=[[r/255, g/255, b/255]])
    ax3d.set_xlabel("R")
    ax3d.set_ylabel("G")
    ax3d.set_zlabel("B")
    ax3d.set_xlim(0, 120)
    ax3d.set_ylim(0, 120)
    ax3d.set_zlim(0, 120)

    # 根據選擇的視角調整顯示
    if view_option == "側視圖":
        ax3d.view_init(elev=30, azim=120)
    elif view_option == "俯視圖":
        ax3d.view_init(elev=90, azim=-90)

    plt.tight_layout()
    st.pyplot(fig_3d)
    plt.close(fig_3d)

    st.subheader("RGB 漸層預覽")
    gradient = np.linspace(1, 13, 500)
    rgb_gradient = np.array([get_rgb(ph) for ph in gradient]) / 255.0
    fig_grad, ax_grad = plt.subplots(figsize=(6, 1))
    ax_grad.imshow([rgb_gradient], aspect='auto')
    ax_grad.axis('off')
    st.pyplot(fig_grad)
    plt.close(fig_grad)
