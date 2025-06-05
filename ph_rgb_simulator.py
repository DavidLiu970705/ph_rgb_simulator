import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ✅ 設定中文字型（依系統情況調整）
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False

# 波長轉 RGB（400~700nm）
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

# 📊 資料
ph_values = list(range(1, 15))
peak_wavelengths = [
    530, 530, 540, 550, 560, 580,
    605, 615, 625, 635, 645,
    425, 425, 425
]
absorb_colors = [wavelength_to_rgb(wl) for wl in peak_wavelengths]

# 🖼️ 主圖：pH 對波長（含 RGB 色點）
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(ph_values, peak_wavelengths, color="gray", linestyle="--", label="平均吸收波長")
for ph, wl, color in zip(ph_values, peak_wavelengths, absorb_colors):
    ax1.scatter(ph, wl, color=color, s=100, edgecolor='black')

ax1.set_title("蝶豆花花青素在不同 pH 下的吸收光線波長與對應色光", fontsize=14)
ax1.set_xlabel("pH 值", fontsize=12)
ax1.set_ylabel("吸收峰值（nm）", fontsize=12)
ax1.grid(True)
ax1.set_xticks(ph_values)
ax1.legend()
fig1.tight_layout()

# 🌈 第二張圖：可見光譜色條
wavelengths = np.linspace(380, 780, 400)
colors = [wavelength_to_rgb(wl) for wl in wavelengths]
colors = np.array(colors).reshape(1, -1, 3)

fig2, ax2 = plt.subplots(figsize=(10, 2))
ax2.imshow(colors, extent=[380, 780, 0, 1], aspect='auto')
ax2.set_yticks([])
ax2.set_xlabel("波長（nm）")
ax2.set_title("可見光波段對應色彩")
fig2.tight_layout()

# 🎨 顯示到網頁上
st.title("🌸 蝶豆花顏色變化與吸收波長分析")
st.markdown("以下展示 pH 值如何影響蝶豆花萃取液的吸收波長與對應可見光顏色：")

st.pyplot(fig1)
st.markdown("接下來是一張可見光波段與色彩的參考圖：")
st.pyplot(fig2)

