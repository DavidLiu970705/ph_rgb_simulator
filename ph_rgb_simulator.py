import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

# 字型設定（顯示中文）
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="pH 與 RGB 對應模擬器", layout="wide")

# --- 波長轉 RGB ---
def wavelength_to_rgb(wavelength):
    gamma = 0.8
    intensity_max = 1
    factor = 0.0
    R = G = B = 0.0

    if 380 <= wavelength <= 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength <= 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength <= 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength <= 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength <= 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 <= wavelength <= 780:
        R = 1.0
        G = 0.0
        B = 0.0

    if 380 <= wavelength <= 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 420 <= wavelength <= 700:
        factor = 1.0
    elif 700 <= wavelength <= 780:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)

    R = round(intensity_max * R * factor, 3)
    G = round(intensity_max * G * factor, 3)
    B = round(intensity_max * B * factor, 3)

    return (R, G, B)

# --- 資料 ---
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

# --- 主頁內容：顯示圖 ---
st.title("pH 與蝶豆花顏色模擬")

# RGB 模擬展示
st.subheader("📷 根據 pH 模擬 RGB 色彩")
ph_range = np.linspace(1, 13, 100)
colors = [get_rgb(p) for p in ph_range]
hex_colors = [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in colors]

fig1, ax1 = plt.subplots(figsize=(10, 1))
for i, color in enumerate(hex_colors):
    ax1.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
ax1.set_xlim(0, 100)
ax1.set_yticks([])
ax1.set_xticks(np.linspace(0, 100, 13))
ax1.set_xticklabels([f"{p:.0f}" for p in np.linspace(1, 13, 13)])
ax1.set_xlabel("pH 值")
st.pyplot(fig1)

# 加入吸收波長圖
st.subheader("🌈 吸收光波長對應色彩圖（實驗觀察）")

ph_list = list(range(1, 15))
peak_wavelengths = [
    530, 530, 540, 550, 560, 580,
    605, 615, 625, 635, 645,
    425, 425, 425
]
absorb_colors = [wavelength_to_rgb(wl) for wl in peak_wavelengths]

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(ph_list, peak_wavelengths, color="gray", linestyle="--", label="平均吸收波長")

for ph, wl, color in zip(ph_list, peak_wavelengths, absorb_colors):
    ax2.scatter(ph, wl, color=color, s=100, edgecolor='black')

ax2.set_title("蝶豆花花青素在不同 pH 下的吸收光線波長與對應色光", fontsize=14)
ax2.set_xlabel("pH 值")
ax2.set_ylabel("吸收峰值（nm）")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)
