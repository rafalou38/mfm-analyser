import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
import matplotlib
import os
import sys

matplotlib.rcParams['lines.markersize'] = 2

file = sys.argv[1] 

df = pd.read_csv(file)



mask_std = (df["phase_std"] < 2) #& (df["t"] > 70)

def bin_average(x, y, step=0.5, y_std=None):
    x_min = np.floor(x.min() / step) * step
    x_max = np.ceil(x.max() / step) * step
    bins = np.arange(x_min, x_max + step, step)
    x_avg, y_avg, y_avg_std = [], [], []

    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i+1]) & mask_std
        if mask.any():
            x_avg.append(bins[i] + step / 2)
            y_avg.append(y[mask].mean())

            if y_std is not None:
                n = mask.sum()
                y_avg_std.append(np.sqrt(np.sum((y_std[mask])**2)) / n)

    if y_std is not None:
        return np.array(x_avg), np.array(y_avg), np.array(y_avg_std)
    return np.array(x_avg), np.array(y_avg)

# plt.scatter(df["t"][mask_std], df["phase"][mask_std])
# plt.scatter(df["t"][mask_std], df["f0"][mask_std])
# plt.scatter(df["t"][mask_std], df["amp"][mask_std])


plt.figure(figsize=(6, 12))
plt.subplot(3, 1, 1)
plt.title("Données Temporelles")
# plt.scatter(df["t"][mask_std], df["phase"][mask_std])

plt.errorbar(df["t"][mask_std], df["phase"][mask_std], yerr=df["phase_std"][mask_std],
            fmt='.',
            capsize=3,
            alpha=0.6,
            color='steelblue',
            ecolor='steelblue',
            label='Data ± std')

plt.grid()
plt.ylabel("Phase (degrees)")


plt.subplot(3, 1, 2)
plt.scatter(df["t"][mask_std], df["amp"][mask_std])
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")



plt.subplot(3, 1, 3)
# plt.gca().yaxis.set_major_locator(MultipleLocator(1))
# plt.gca().yaxis.set_minor_locator(MultipleLocator(0.2))
plt.grid()
# plt.grid(True, 'minor', color="#f0f0f0") 


x_binned, f_binned = bin_average(df["t"], df["f0"], step=4)

f_fit_params1 = np.polyfit(df["t"][mask_std], df["f0"][mask_std], deg=1)
f_fit1 = f_fit_params1[0] * df["t"][mask_std] + f_fit_params1[1]

f_fit_params2 = np.polyfit(df["t"][mask_std], df["f0"][mask_std], deg=2)
f_fit2 = f_fit_params2[0] * df["t"][mask_std]**2 + f_fit_params2[1] * df["t"][mask_std] + f_fit_params2[2]

plt.scatter(df["t"][mask_std], df["f0"][mask_std], alpha=0.5)
plt.plot(df["t"][mask_std], f_fit1, label="Fit 1", color="red")
plt.plot(df["t"][mask_std], f_fit2, label="Fit 2", color="green")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")


plt.savefig("phase_draw_01.png", dpi=300)

# print(df["f0"].shape)
# print(f_fit.shape)

# # # plt.subplot(3, 2, 2)
plt.figure(figsize=(6, 12))
plt.subplot(2, 1, 1)
plt.title("Lissage fréquence")
# f_binned, p_binned, p_binned_std = bin_average(f_fit, df["phase"], step=0.02, y_std=df["phase_std"])
# plt.scatter(f_binned, p_binned)
plt.scatter(f_fit1, df["phase"][mask_std], color="red", label="deg 1", alpha=0.05)
plt.scatter(f_fit2, df["phase"][mask_std], color="green", label="deg2")
plt.scatter(df["f0"][mask_std], df["phase"][mask_std], alpha=0.05, label="raw")
# plt.errorbar(f_binned, p_binned, yerr=p_binned_std, fmt='.', capsize=3)
plt.grid()
plt.ylabel("Déphasage (°)")


plt.subplot(2, 1, 2)
# f_binned, a_binned = bin_average(f_fit, df["amp"], step=0.02)
# plt.scatter(f_binned, a_binned)

# plt.gca().yaxis.set_minor_locator(MultipleLocator(0.2))
plt.scatter(f_fit1, df["amp"][mask_std], color="red", label="deg 1", alpha=0.05)
plt.scatter(f_fit2, df["amp"][mask_std], color="green", label="deg 2")
plt.scatter(df["f0"][mask_std], df["amp"][mask_std], alpha=0.05, label="raw")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")

plt.gca().yaxis.set_minor_locator(MultipleLocator(0.2))
plt.savefig("phase_draw_02.png", dpi=300)




plt.figure(figsize=(6, 12))
plt.subplot(2, 1, 1)
plt.title("Données Brutes")

plt.scatter(df["f0"][mask_std], df["phase"][mask_std], alpha=0.5, label="Data")
plt.grid()
plt.ylabel("Déphasage (°)")
plt.subplot(2, 1, 2)

plt.scatter(df["f0"][mask_std], df["amp"][mask_std], alpha=0.5, label="Data")
plt.ylabel("Amplitude")
plt.xlabel("Fréquence (Hz)")
plt.grid()
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.2))
plt.savefig("phase_draw_03.png", dpi=300)

# plt.errorbar(df["t"][mask_std], df["phase"][mask_std], yerr=df["phase_std"][mask_std],
#             fmt='.',           # marqueurs ronds
#             capsize=3,         # petits chapeaux sur les barres
#             alpha=0.6,
#             color='steelblue',
#             ecolor='steelblue',
#             label='Data ± std')



plt.grid()
plt.legend()
plt.show()
