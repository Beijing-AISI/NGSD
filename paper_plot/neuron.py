import math
import matplotlib.pyplot as plt

FILES = {
    "GCG": {
        "EMA": "./gate/ema_gcg_log.txt",
        "SMG": "./gate/adam_gcg_log.txt",
        "Neuron": "./gate/lif_gcg_log.txt",
    },
    "PAIR": {
        "EMA": "./gate/ema_pair_log.txt",
        "SMG": "./gate/adam_pair_log.txt",
        "Neuron": "./gate/lif_pair_log.txt",
    },
    "AutoDAN": {
        "EMA": "./gate/ema_autodan_log.txt",
        "SMG": "./gate/adam_autodan_log.txt",
        "Neuron": "./gate/lif_autodan_log.txt",
    },
}

COLORS = {
    "EMA": "#b8dbb3",
    "SMG": "#94c6cd",

    "Neuron": "#eab883",
}

N = 150  # 取前150条

def read_first_n_floats(path: str, n: int):
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(vals) >= n:
                break
            s = line.strip()
            if not s:
                continue
            try:
                x = float(s)
            except ValueError:
                # 如果某行不是纯数字，跳过
                continue
            if not math.isfinite(x):
                continue
            vals.append(x)
    return vals

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, (group_name, group_files) in zip(axes, FILES.items()):
    for name, path in group_files.items():
        ys = read_first_n_floats(path, N)
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, label=name, color=COLORS.get(name))

    ax.set_title(group_name)
    ax.set_xlabel("Tokens", fontsize=13)

    # 仅保留横向 grid（y 方向）
    ax.grid(True, axis="y", alpha=0.3)

    # 去掉上、右边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Value", fontsize=13)
# 将特定 y 轴刻度标红
highlight_ticks = {0.5, 0.75, 1.0}

for tick, label in zip(axes[0].get_yticks(), axes[0].get_yticklabels()):
    if tick in highlight_ticks:
        label.set_color("red")

axes[0].legend()

plt.tight_layout()
out_path = "neuron_robustness.pdf"
plt.savefig(out_path, dpi=200)
plt.show()

print("Saved plot to:", out_path)