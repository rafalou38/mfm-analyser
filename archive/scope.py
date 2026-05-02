import pyvisa
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.signal as signal
from scipy.optimize import curve_fit


# -------------------------
# CONFIG
# -------------------------
SCOPE_IP = "192.168.1.42"
CHANNEL = "CHAN1"
UPDATE_DELAY = 1/1   # seconds between refresh (~1 Hz)

# -------------------------
# CONNECT TO SCOPE
# -------------------------
rm = pyvisa.ResourceManager()
scope = rm.open_resource(f"TCPIP0::{SCOPE_IP}::INSTR")
scope.timeout = 5000

print(scope.query("*IDN?"))

# -------------------------
# CONFIGURE WAVEFORM MODE
# -------------------------
scope.write(":WAV:FORM BYTE")
# scope.write(":WAV:MODE RAW")
scope.write(":WAV:MODE NORM")
scope.write(f":WAV:SOUR {CHANNEL}")

def get_waveform(channel, raw=False):
    scope.write(f":WAV:SOUR {channel}")
    scope.write(":WAV:FORM BYTE")

    if raw:
        scope.write(":WAV:MODE RAW")

        points = int(scope.query(":WAV:POIN?"))
        scope.write(":WAV:STAR 1")
        scope.write(f":WAV:STOP {points}")
    else:
        scope.write(":WAV:MODE NORM")

    values = scope.query_binary_values(":WAV:DATA?", datatype='B', container=np.array)

    y_inc = float(scope.query(":WAV:YINC?"))
    y_origin = float(scope.query(":WAV:YOR?"))
    y_ref = float(scope.query(":WAV:YREF?"))

    x_inc = float(scope.query(":WAV:XINC?"))
    x_origin = float(scope.query(":WAV:XOR?"))
    x_ref = float(scope.query(":WAV:XREF?"))

    volts = (values - y_ref) * y_inc + y_origin
    time_axis = (np.arange(len(volts)) - x_ref) * x_inc + x_origin

    return time_axis, volts

# -------------------------
# SETUP PLOT
# -------------------------
plt.ion()
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 6))

line1, = axes[0, 0].plot([], [])
line1_fit, = axes[0, 0].plot([], [], color='red', linestyle='--')
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Voltage (V)")
axes[0, 0].set_title("CH1")

vlines1 = axes[0,0].vlines([], ymin=0, ymax=1, color='red', linestyle='--')


line2, = axes[1, 0].plot([], [])
line2_fit, = axes[1, 0].plot([], [], color='red', linestyle='--')
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Voltage (V)")
axes[1, 0].set_title("CH2")

vlines2 = axes[1,0].vlines([], ymin=0, ymax=1, color='red', linestyle='--')

line3, = axes[0, 1].plot([], [])
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Dphi (degrees)")
axes[0, 1].set_title("Phase Shift")

line4, = axes[1, 1].plot([], [])
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Voltage (V)")
axes[1, 1].set_title("Amplitude")

# Dots
line5, = axes[2, 1].plot([], [], linestyle='None', marker='o')
line5_avg, = axes[2, 1].plot([], [], color='green')

axes[2, 1].set_xlabel("Time (s)")
axes[2, 1].set_ylabel("Dphi (degrees)")
axes[2, 1].set_title("Phase Shift (peaks)")


# lineS1, = axes[1, 1].plot([], [], color='green')




peaks_shift = []
peaks_shift_avg = []
peaks_shift_t = []

shift_values = []
amplitude_values = []
times = []


# Modèle
def sine(t, A, f, phi, C):
    return A * np.sin(2*np.pi*f*t + phi) + C


# -------------------------
# REAL-TIME LOOP
# -------------------------
try:
    while True:

        t0 = time.time()

        time_div = float(scope.query(":TIM:SCAL?"))
        window_time = time_div * 10

        # print(f"Time div: {time_div:.3e} s/div, Window: {window_time:.3e} s")

        scope.write(":STOP")
        t1, v1 = get_waveform("CHAN1", raw=False)
        t2, v2 = get_waveform("CHAN2", raw=False)
        scope.write(":RUN")
        time.sleep(window_time)

        if len(t1) == 0 or len(t2) == 0:
            print("Empty waveform")
            continue

        params1, _ = curve_fit(sine, t1, v1, p0=[10, 35, 0,0])
        A_fit, f_fit, phi_fit, C_fit = params1
        # print("A =", A_fit)
        # print("f =", f_fit)
        # print("phi =", phi_fit)
        v1_fit = sine(t1, *params1)

        params2, _ = curve_fit(sine, t2, v2, p0=[10, 35, 0,0])
        A_fit2, f_fit2, phi_fit2, C_fit2 = params2
        # print("A =", A_fit2)
        # print("f =", f_fit2)
        # print("phi =", phi_fit2)
        v2_fit = sine(t2, *params2)

        shift = phi_fit - phi_fit2
        shift_values.append(shift)

        ampl = max(v1) - min(v1)
        amplitude_values.append(ampl)

        times.append(t0)
        line3.set_data(times, shift_values)
        axes[0, 1].set_xlim(times[0], times[-1])
        axes[0, 1].set_ylim(np.min(shift_values), np.max(shift_values))

        line4.set_data(times, amplitude_values)
        axes[1, 1].set_xlim(times[0], times[-1])
        axes[1, 1].set_ylim(np.min(amplitude_values), np.max(amplitude_values))


        # Peaks

        mins1, _ = signal.find_peaks(-v1, prominence=0.1, distance=5)
        print(f"CH1 Peaks: {len(mins1)}", t1[mins1])

        mins2, _ = signal.find_peaks(-v2_fit, prominence=0.1, distance=5)

        print(f"CH2 Peaks: {len(mins2)}", t2[mins2])

        vlines1.set_segments([[(t1[m], np.min(v1)), (t1[m], np.max(v1))] for m in mins1])
        vlines2.set_segments([[(t2[m], np.min(v2_fit)), (t2[m], np.max(v2_fit))] for m in mins2])

        # mins2, _ = signal.find_peaks(-v2, prominence=)

        # plt.vlines(data['t'].iloc[mins], ymin=m, ymax=M, color='red', linestyle='--', label='Minima')

        # axes[0, 0].vlines(t1[mins1], ymin=np.min(v1), ymax=np.max(v1), color='red', linestyle='--', label='Minima')
        # axes[1, 0].vlines(t2[mins2], ymin=np.min(v2), ymax=np.max(v2), color='red', linestyle='--', label='Minima')

        sp = 0

        for m1, m2 in zip(mins1, mins2):
            # print(f"Peak shift: {t1[m1] - t2[m2]:.3e} s at t={t0 + (t1[m1] + t2[m2])/2:.3e} s")
            dp = t1[m1] - t2[m2]
            t = t0 + (t1[m1] + t2[m2]) / 2

            sp += dp

            peaks_shift.append(dp)
            peaks_shift_t.append(t)

        for i in range(len(mins1)):
            peaks_shift_avg.append(sp/len(mins1))

        print(len(peaks_shift), len(peaks_shift_avg))

        line5.set_data(peaks_shift_t, peaks_shift)
        line5_avg.set_data(peaks_shift_t, peaks_shift_avg)

        axes[2, 1].set_xlim(peaks_shift_t[0], peaks_shift_t[-1])
        axes[2, 1].set_ylim(np.min(peaks_shift), np.max(peaks_shift))


        # sync = sine(t2, 1, f_fit2, phi_fit2, 0) * v1

        # sos = signal.butter(4, 2, 'low', fs=1000, output='sos')
        # I = signal.sosfiltfilt(sos, sync)

        # lineS1.set_data(t1, I)
        # axes[1, 1].set_xlim(t1[0], t1[-1])
        # axes[1, 1].set_ylim(np.min(I), np.max(I))

        line1.set_data(t1, v1)
        line1_fit.set_data(t1, v1_fit)
        line2.set_data(t2, v2)
        line2_fit.set_data(t2, v2_fit)

        axes[0, 0].set_xlim(t1[0], t1[-1])
        axes[0, 0].set_ylim(np.min(v1), np.max(v1))

        axes[1, 0].set_xlim(t2[0], t2[-1])
        axes[1, 0].set_ylim(np.min(v2), np.max(v2))

        fig.canvas.draw()
        fig.canvas.flush_events()


        print("took", time.time() - t0)

        # time.sleep(UPDATE_DELAY)

except KeyboardInterrupt:
    print("Stopped.")

finally:
    scope.close()
