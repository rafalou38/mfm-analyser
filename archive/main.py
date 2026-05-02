import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def plot(f):
    data = pd.read_csv(f)
    t0 = float(data.columns[2].split('=')[1].strip())
    tInc = float(data.columns[3].split('=')[1].strip())
    data['t'] = t0 + data.index * tInc

    plt.figure(figsize=(10, 6))
    plt.plot(data['t'], data['CH1V'], label='CH1V')
    plt.plot(data['t'], data['CH2V'], label='CH2V')


    mins, _ = find_peaks(-data['CH1V'], prominence=0.1)
    # plt.plot(data['t'].iloc[mins], data['CH1V'].iloc[mins], "ro")

    m = min(data['CH1V'].min(), data['CH2V'].min())
    M = max(data['CH1V'].max(), data['CH2V'].max())

    plt.vlines(data['t'].iloc[mins], ymin=m, ymax=M, color='red', linestyle='--', label='Minima')

    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage vs Time')
    plt.legend()
    plt.grid()

if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
    print("Usage: python main.py <csv_file>")
else:
    plot(sys.argv[1])
    plt.savefig(os.path.splitext(sys.argv[1])[0] + '.png')
    plt.show()
