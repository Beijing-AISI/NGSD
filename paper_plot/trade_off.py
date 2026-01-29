import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

label_fontsize = 10
label_fontweight = "normal"

def pick_font(candidates):
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"  # safe fallback always shipped with matplotlib


label_fontfamily = "DejaVu Sans"

# -----------------------------
# Data
# -----------------------------
alpha = [0, 0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9]
usrbenign = [79.20, 75.23, 74.98, 75.06, 68.41, 59.39, 56.44, 53.24]
gsm8k = [27.1, 24.9, 20.20, 19.71, 16.10, 7.28, 6.97, 6.29]
safe = [4.50, 4.66, 4.54, 3.90, 2.26, 1.82, 1.54, 1.34]
safe_asr = ["76%", "72%", "76%", "60%", "22%", "16%", "8%", "2%"]

# Treat alpha as categorical (equal spacing)
x = np.arange(len(alpha))
x_labels = [str(a) for a in alpha]

baseline_color = "#cc7c71"  # neutral gray for baseline

# -----------------------------
# Single-panel figure
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 4))

# Border / spine styling
border_color = "#9f9f9f"
border_lw = 1  # make it thicker so the change is visible
for spine in ax.spines.values():
    spine.set_color(border_color)
    spine.set_linewidth(border_lw)



# Main left y-axis: USRBenign
l1, = ax.plot(
    x,
    usrbenign,
    linewidth=1.8,
    label="USRBenign",
    color="#faac8c",
    zorder=3,
)
# Markers for USRBenign (baseline highlighted)
for i, (xi, yi) in enumerate(zip(x, usrbenign)):
    ax.scatter(
        xi,
        yi,
        marker="o",
        s=(3 * 3) ** 2 / 4,
        facecolors=ax.get_facecolor(),
        edgecolors="#8d2f25" if i == 0 else "#faac8c",
        linewidths=0.9,
        zorder=4,
    )
ax.set_ylabel("FalseRej-USR(%)",
              fontsize=label_fontsize,
              fontfamily=label_fontfamily,
              fontweight=label_fontweight,
              )
ax.set_ylim(50, 85)
# Match left y-axis color with USRBenign line
ax.yaxis.label.set_color("#faac8c")
ax.tick_params(axis="y", colors="#faac8c")
ax.spines["left"].set_color("#faac8c")


# Right y-axis: GSM8K
ax_r = ax.twinx()

# IMPORTANT: the right spine belongs to ax_r (twinx), so style it too
for spine in ax_r.spines.values():
    spine.set_color(border_color)
    spine.set_linewidth(border_lw)


l2, = ax_r.plot(
    x,
    gsm8k,
    linewidth=1.8,
    label="GSM8K",
    color="#c497b2",
    zorder=3,
)
# Markers for GSM8K (baseline highlighted)
for i, (xi, yi) in enumerate(zip(x, gsm8k)):
    ax_r.scatter(
        xi,
        yi,
        marker="^",
        s=(3 * 3) ** 2 / 4,
        facecolors=ax.get_facecolor(),
        edgecolors="#8d2f25" if i == 0 else "#c497b2",
        linewidths=1.4,
        zorder=4,
    )
ax_r.set_ylabel("GSM8K-ROUGE(%)",
                fontsize=label_fontsize,
                fontfamily=label_fontfamily,
                fontweight=label_fontweight,
                )
ax_r.set_ylim(5, 30)
# Match right y-axis color with GSM8K line
ax_r.yaxis.label.set_color("#c497b2")
ax_r.tick_params(axis="y", colors="#c497b2")
ax_r.spines["right"].set_color("#c497b2")

# X axis
ax.set_xlabel("SafeDecoding Alpha",
              fontsize=label_fontsize,
              fontfamily=label_fontfamily,
              fontweight=label_fontweight,
              )
ax.set_xticks(x)
ax.set_xticklabels(x_labels)

# -----------------------------
# Safe bars mapped to visual band (no extra axis)
# -----------------------------
ymin, ymax = ax.get_ylim()
band_height = 0.15 * (ymax - ymin)

# Map Safe proportionally: height = safe / 5.5 * band_height
safe_ref = 5.5  # reference maximum for normalization
safe_vis = [
     (s / safe_ref) * band_height * 5
    for s in safe
]

# Safe bars (baseline highlighted)
bar_colors = [baseline_color] + ["#9ac9db"] * (len(x) - 1)
ax.bar(
    x,
    safe_vis,
    bottom=ymin,
    width=0.6,
    color=bar_colors,
    alpha=0.7,
    zorder=0
)

# Optional: add a small in-plot label for Safe (no axis)
for i, (xi, s, s_a, h, a) in enumerate(zip(x, safe, safe_asr, safe_vis, alpha)):
    ax.text(
        xi,
        ymin + h - 0.15,              # slightly inside the bar
        f"\n{s:.2f}\n(" + s_a + ")",
        ha="center",
        va="top",
        fontsize=7,
        color="#8d2f25" if i == 0 else "#14517c"
    )

# Legend combining both lines
ax.legend(handles=[l1, l2], loc="upper right", frameon=False,
          prop={
              "family": label_fontfamily,
              "size": 10,
              "weight": "normal",
          }
          )

ax.annotate(
    "No Defense",
    xy=(0.1, 82),          # 箭头指向的位置（x 是第 4 个 alpha，因为你用的是 np.arange）
    xytext=(0.5, 83),    # 文字位置
    fontsize=10,
    fontfamily=label_fontfamily,
    color="#8d2f25",
    arrowprops=dict(
        arrowstyle="->",
        lw=1.2,
        color="#8d2f25",
    ),
)
ax.axvline(
    x=2.5,                 # 竖线所在的 x 位置（数据坐标）
    color="#9ac9db",      # 灰色
    linewidth=1.0,
    linestyle="--",       # 可选：虚线
    zorder=1
)

ax.annotate(
    "Marginal Effect Exists",
    xy=(2.2, 82),          # 箭头指向的位置（x 是第 4 个 alpha，因为你用的是 np.arange）
    xytext=(2.8, 81.7),    # 文字位置
    fontsize=10,
    fontfamily=label_fontfamily,
    color="#333333",
    arrowprops=dict(
        arrowstyle="->",
        lw=1,
        color="#333333",
    ),
)

plt.tight_layout()
plt.savefig("trade_off.pdf")
plt.show()