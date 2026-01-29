import numpy as np
import matplotlib.pyplot as plt

# 读取日志文件
v = np.loadtxt('v_log.txt')
mag = np.loadtxt('v_adam_log.txt')

# 只取前150个 sample
N = 150
v = v[:N]
mag = mag[:N]
x = np.arange(N)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(x, v, label='LIF')
plt.plot(x, mag, label='Adam')

# 阈值线
plt.axhline(1.0, linestyle='--', label='LIF threshold = 1')
plt.axhline(0.5, linestyle='--', label='Adam threshold = 0.5', color='orange')


# 仅取消上、右边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Sample index')
plt.title('nueron vs. adam')
plt.grid(axis='y')
plt.legend()
plt.tight_layout()
plt.show()