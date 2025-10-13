import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

np.random.seed(42)
n = 20000       # больше данных
r = 0.5
tau = np.arange(50)

x = np.zeros(n)
for t in range(1, n):
    x[t] = r * x[t - 1] + np.random.normal(0, 0.3)  # меньше шум

emp_acf = acf(x, nlags=49, fft=True)
theor_acf = r ** tau

plt.figure(figsize=(8, 6))
plt.plot(tau, theor_acf, 'r-', linewidth=2, label='Теоретическая ACF')
plt.plot(tau, emp_acf, 'bo-', markersize=3, label='Эмпирическая ACF')
plt.title('ACF для экспоненциально коррелированного ряда (r=0.5)', fontsize=12, fontweight='bold')
plt.xlabel('Лаг (τ)')
plt.ylabel('Автокорреляционный коэффициент')
plt.legend()
plt.grid(True)
plt.show()