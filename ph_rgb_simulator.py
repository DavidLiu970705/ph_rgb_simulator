import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import io

st.set_page_config(page_title="pH 與 RGB 對應模擬器", layout="wide")

# --- 原始資料：兩組 ---
ph_data_dict = {
    "垂拍": {
        "ph": np.array([1.4, 1.4, 2.4, 2.4, 3.4, 3.4, 4.4, 4.4, 7, 7, 9, 9, 10, 10, 11, 11, 12, 12]),
        "r":  np.array([111, 110, 78, 73, 46, 52, 30, 32, 50, 59, 29, 39, 30, 26, 44, 42, 35, 47]),
        "g":  np.array([39, 38, 35, 28, 33, 42, 49, 50, 70, 74, 47, 55, 48, 44, 71, 70, 37, 51]),
        "b":  np.array([59, 58, 62, 57, 63, 75, 63, 64, 81, 79, 59, 70, 60, 58, 78, 74, 23, 37])
    },
    "側拍": {
        "ph": np.array([1.4, 2.4, 3.4, 4.4, 7, 9, 10, 11, 12]),
        "r":  np.array([140, 132, 126, 112, 124, 121, 120, 110, 126]),
        "g":  np.array([114, 117, 124, 122, 128, 126, 129, 123, 128]),
        "b":  np.array([117, 122, 127, 124, 131, 130, 134, 129, 104])
    }
}

def poly2(x, a, b, c):
    return a * x**2 + b * x + c

def center_axis(data, margin=30):
    center = (np.max(data) + np.min(data)) / 2
    span = (np.max(data) - np.min(data)) / 2 + margin
    return center - span, center + span

# --- 側邊選單 ---
with st.sidebar:
    st.title("pH 控制面板")
    dataset = st.selectbox("選擇資料來源", ["垂拍", "側拍"])
    ph_input = st.slider("選擇 pH 值", 1.0, 13.0, 7.0, step=0.01)

    data = ph_data_dict[dataset]
    ph_values, r_values, g_values, b_values = data["ph"], data["r"], data["g"], data["b"]

    # 模型擬合
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

    # 漸層色條
    ph_range = np.linspace(1, 13, 100)
    gradient = [
        f'rgb({int(np.clip(poly2(p, *params_r), 0, 255))},'
        f'{int(np.clip(poly2(p, *params_g), 0, 255))},'
        f'{int(np.clip(poly2(p, *params_b), 0, 255))})'
        for p in ph_range
    ]
    gradient_css = ','.join(gradient)
    st.markdown(
        f"""
        <div style="width:100%;height:30px;
                    background: linear-gradient(to right, {gradient_css});
                    border-radius:6px;border:1px solid #aaa;margin-top:5px;">
        </div>
        """,
        unsafe_allow_html=True
    )

# --- 主內容 ---
col1, col2 = st.columns(2)

# --- 2D 曲線圖 ---
with col1:
    st.subheader("pH 對 RGB 曲線圖")
    x_plot = np.linspace(1, 13, 200)
    fig2d, ax = plt.subplots()
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
    st.pyplot(fig2d)

    # 匯出 2D 圖
    buffer2d = io.BytesIO()
    fig2d.savefig(buffer2d, format='png')
    st.download_button("下載 2D 曲線圖", data=buffer2d.getvalue(), file_name="ph_rgb_2d.png", mime="image/png")
    plt.close(fig2d)

# --- 3D 圖 ---
with col2:
    st.subheader("RGB 三維分布圖")
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')

    r_curve = poly2(x_plot, *params_r)
    g_curve = poly2(x_plot, *params_g)
    b_curve = poly2(x_plot, *params_b)

    ax3d.plot(r_curve, g_curve, b_curve, color="black", label="RGB 曲線")
    ax3d.scatter(r_values, g_values, b_values,
                 c=np.stack([r_values, g_values, b_values], axis=1)/255.0,
                 s=40, label='資料點')
    ax3d.scatter(r, g, b, s=100, c=[[r/255, g/255, b/255]], edgecolors='black', label='目前點')

    r_min, r_max = center_axis(np.concatenate([r_values, [r]]))
    g_min, g_max = center_axis(np.concatenate([g_values, [g]]))
    b_min, b_max = center_axis(np.concatenate([b_values, [b]]))

    ax3d.set_xlim(r_min, r_max)
    ax3d.set_ylim(g_min, g_max)
    ax3d.set_zlim(b_min, b_max)
    ax3d.set_xlabel("R")
    ax3d.set_ylabel("G")
    ax3d.set_zlabel("B")
    ax3d.set_box_aspect([
        r_max - r_min,
        g_max - g_min,
        b_max - b_min
    ])
    plt.tight_layout()
    st.pyplot(fig3d)

    # 匯出 3D 圖
    buffer3d = io.BytesIO()
    fig3d.savefig(buffer3d, format='png')
    st.download_button("下載 3D 分布圖", data=buffer3d.getvalue(), file_name="ph_rgb_3d.png", mime="image/png")
    plt.close(fig3d)

# --- 匯出 CSV ---
st.download_button("匯出目前 RGB 成 CSV",
                   data=f"pH,R,G,B\n{ph_input},{r},{g},{b}",
                   file_name="ph_rgb.csv",
                   mime="text/csv")

