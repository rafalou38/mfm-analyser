import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def plot(f):
    data = pd.read_csv(f)
    t0 = float(data.columns[1].split('=')[1].strip())
    tInc = float(data.columns[2].split('=')[1].strip())
    data['t'] = t0 + data.index * tInc

    plt.figure(figsize=(50, 5))
    plt.plot(data['t'][450000:660000], data['CH1V'][450000:660000], label='CH1V')

    # mins, _ = find_peaks(-data['CH1V'], prominence=0.1)
    # plt.plot(data['t'].iloc[mins], data['CH1V'].iloc[mins], "ro")


    # plt.vlines(data['t'].iloc[mins], ymin=min(data['CH1V']), ymax=max(data['CH1V']), color='red', linestyle='--', label='Minima')

    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage vs Time')
    plt.legend()
    plt.grid()

if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
    print("Usage: python main.py <csv_file>")
else:
    plot(sys.argv[1])
    plt.savefig(os.path.splitext(sys.argv[1])[0] + '.png', dpi=300)
    plt.show()
