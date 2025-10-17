import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

np.random.seed(42)
n = 20000       # больше данных
r = 0.5
tau = np.arange(11)

x = np.zeros(n)
for t in range(1, n):
    x[t] = r * x[t - 1] + np.random.normal(0, 0.3)  # меньше шум

emp_acf = acf(x, nlags=10, fft=True)
theor_acf = r ** tau

plt.figure(figsize=(8, 6))
plt.plot(tau, theor_acf, 'r-', linewidth=2, label='Теоретическая ACF')
plt.plot(tau, emp_acf, 'bo-', markersize=5, label='Эмпирическая ACF')
plt.title('ACF для экспоненциально коррелированного ряда (r=0.5)', fontsize=12, fontweight='bold')
plt.xlabel('Лаг (τ)')
plt.ylabel('Автокорреляционный коэффициент')
plt.legend()
plt.grid(True)
plt.show()

# Данные
r = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9])
E_delta_1_25 = np.array([1.000, 0.983, 0.898, 0.781, 0.681, 0.645, 0.590])
E_delta_2 = np.array([1.000, 0.992, 0.953, 0.894, 0.818, 0.766, 0.675])


# Построение графика
plt.figure(figsize=(8, 5))
plt.plot(r, E_delta_1_25, 'r-o', label='δ = 1,25', linewidth=2, markersize=6)
plt.plot(r, E_delta_2, 'b-s', label='δ = 2', linewidth=2, markersize=6)

# Настройка осей
plt.xticks(np.arange(0, 1.0, 0.1))
plt.yticks(np.arange(0.5, 1.05, 0.05))

# Оформление
plt.title("Усредненный показатель относительной эффективности", fontsize=13, fontweight='bold')
plt.xlabel("Коэффициентом авторегрессии, r", fontsize=10)
plt.ylabel("Показатель относительной эффективности", fontsize=10, rotation=90, labelpad=10)

# Только основная сетка
plt.grid(True, linestyle='-', color='gray', alpha=0.6)

plt.legend(loc='upper right', fontsize=11)
plt.ylim(0.5, 1.05)
plt.xlim(0, 0.95)
plt.tight_layout()
plt.show()

# Линейнейная аппроксимация

# === 1. Линейная аппроксимация (без фиксации на (0, 1)) ===
a1, b1 = np.polyfit(r, E_delta_1_25, 1)
a2, b2 = np.polyfit(r, E_delta_2, 1)

# === 2. Линии аппроксимации ===
r_fit = np.linspace(0, 1.2, 200)
fit_1_25 = a1 * r_fit + b1
fit_2 = a2 * r_fit + b2

# === 3. Построение графика ===
plt.figure(figsize=(8, 5))

# Контрольные точки + сплошные линии
plt.plot(r, E_delta_1_25, 'ro', label='δ = 1,25: контрольные значения')
plt.plot(r, E_delta_2, 'bs', label='δ = 2: контрольные значения')
plt.plot(r_fit, fit_1_25, 'r-', linewidth=2, label=f'δ = 1,25: E={a1:.3f}r+{b1:.3f}')
plt.plot(r_fit, fit_2, 'b-', linewidth=2, label=f'δ = 2: E={a2:.3f}r+{b2:.3f}')

# Оформление
plt.title("Усредненный показатель относительной эффективности", fontsize=13, fontweight='bold')
plt.xlabel("Коэффициент авторегрессии, r", fontsize=10)
plt.ylabel("Показатель относительной эффективности", fontsize=10, rotation=90, labelpad=10)
plt.xticks(np.arange(0, 1.21, 0.1))
plt.yticks(np.arange(0.5, 1.05, 0.05))
plt.grid(True, linestyle='-', color='gray', alpha=0.6)
plt.legend(fontsize=10, loc='upper right')
plt.ylim(0.5, 1.02)
plt.xlim(0, 1.2)
plt.tight_layout()
plt.show()
