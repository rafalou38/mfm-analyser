import serial
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import time
import subprocess
from scipy.signal import butter, filtfilt, sosfilt, sosfilt_zi

class LockIn:
    def __init__(self, history):
        self.ptr = 0
        self.phi_ptr = 0

        self.proc = subprocess.Popen(
            ["./serial"],
            stdout=subprocess.PIPE,
            bufsize=0
        )

        self.FRAME_BYTES = 256 * (4 + 4 + 1 + 4)
        self.history_sz = history
        self.phi_history_sz = history * 10

        self.buf0 = np.zeros(self.history_sz, dtype=np.float64)
        self.buf1 = np.zeros(self.history_sz, dtype=np.float64)
        self.bufz = np.zeros(self.history_sz, dtype=np.int8)
        self.buft = np.zeros(self.history_sz, dtype=np.float64)

        self.last_phi = 0.0
        self.phase_acc = 0.0
        self.last_edge_distance = None
        self.bufphi = np.zeros(self.phi_history_sz, dtype=np.float32)
        self.bufamp = np.zeros(self.phi_history_sz, dtype=np.float32)
        self.bufphit = np.zeros(self.phi_history_sz, dtype=np.float32)

        self.x_min = 0.0
        self.x_max = 0.01
        self.y_min = 0.0
        self.y_max = 0.01

        self.ready = False

        self.sos = None
        self.zi_I = None
        self.zi_Q = None

    
    def read_serial(self, handle_phase=True):
        data = self.proc.stdout.read(self.FRAME_BYTES)
        if len(data) != self.FRAME_BYTES:
            print("End of stream")
            exit(1)
        dtype = np.dtype([
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.int8),
            ("t", np.uint32),
        ])
        samples = np.frombuffer(data, dtype=dtype)

        x_values = samples['x']
        y_values = samples['y']
        z_values = samples['z']
        t_values = samples['t'] * 1e-6

        # Process window
        self.update_bounds()
        Ts = abs(np.median(np.diff(t_values)))
        fs = 1 / Ts

        if self.sos is None:
            self.sos = butter(3, 5, fs=fs, output='sos')
            n_sections = self.sos.shape[0]

            self.zi_I = np.zeros((n_sections, 2))
            self.zi_Q = np.zeros((n_sections, 2))
            print("Sampeling frequency: ", fs)
        
        n = len(x_values)
        if self.ptr + n < self.history_sz:
            self.buf0[self.ptr:self.ptr+n] = x_values
            self.buf1[self.ptr:self.ptr+n] = y_values
            self.bufz[self.ptr:self.ptr+n] = z_values
            self.buft[self.ptr:self.ptr+n] = t_values
        else:
            self.ready = True
            k = self.history_sz - self.ptr
            self.buf0[self.ptr:] = x_values[:k]
            self.buf1[self.ptr:] = y_values[:k]
            self.bufz[self.ptr:] = z_values[:k]
            self.buft[self.ptr:] = t_values[:k]

            self.buf0[:n-k] = x_values[k:]
            self.buf1[:n-k] = y_values[k:]
            self.bufz[:n-k] = z_values[k:]
            self.buft[:n-k] = t_values[k:]

        self.ptr = (self.ptr + n) % self.history_sz

        if not handle_phase: return

        edges = np.where((z_values[:-1] == 0) & (z_values[1:] == 1))[0]
        # Hard sync
        phase = np.zeros_like(t_values)

        # estimate frequency from edge spacing if possible
        if len(edges) >= 2:
            periods = np.diff(edges)
            mean_period = np.average(periods)
            print("Current frequency: ", 1/(mean_period) * fs, len(edges))
            dphi = 2 * np.pi / mean_period
            self.last_dphi = dphi
        elif hasattr(self, "last_dphi"):
            dphi = self.last_dphi
        else:
            # fallback
            dphi = 2 * np.pi * 40.0 / fs

        # continuous oscillator
        for i in range(len(phase)):
            if i in edges:
                self.phase_acc = 0.0

            phase[i] = self.phase_acc
            self.phase_acc += dphi

        s = np.sin(phase)
        c = np.cos(phase)

        I_raw = x_values * c
        Q_raw = x_values * s

        I, self.zi_I = sosfilt(self.sos, I_raw, zi=self.zi_I)
        Q, self.zi_Q = sosfilt(self.sos, Q_raw, zi=self.zi_Q)

        phi = np.arctan2(Q, I)
        phi = np.unwrap(
            np.concatenate(([self.last_phi], phi))
        )[1:]
        self.last_phi = phi[-1]

        amp = np.sqrt(I * I + Q * Q)
        print("avg phase: ", np.mean(phi))
        # print(amp)

        # Store RAW data
        if self.ready:
            phi_n = len(phi)
            if self.phi_ptr + phi_n < self.phi_history_sz:
                self.bufamp[self.phi_ptr:self.phi_ptr+phi_n] = amp
                self.bufphi[self.phi_ptr:self.phi_ptr+phi_n] = phi
                self.bufphit[self.phi_ptr:self.phi_ptr+phi_n] = t_values
            else:
                k = self.phi_history_sz - self.phi_ptr
                self.bufamp[self.phi_ptr:] = amp[:k]
                self.bufphi[:phi_n-k] = phi[k:]
                self.bufphit[self.phi_ptr:] = t_values[:k]
                self.bufphit[:phi_n-k] = t_values[k:]

            self.phi_ptr = (self.phi_ptr + phi_n) % self.phi_history_sz

    def update_bounds(self):
        self.x_min = min(self.buf0.min(), self.x_min)
        self.x_max = max(self.buf0.max(), self.x_max)
        self.y_min = min(self.buf1.min(), self.y_min)
        self.y_max = max(self.buf1.max(), self.y_max)

        # Reset if values become too large
        if not np.isfinite([self.x_min, self.x_max, self.y_min, self.y_max]).all() or max(abs(self.x_min), abs(self.x_max), abs(self.y_min), abs(self.y_max)) > 1e6:
            self.x_min = self.buf0.min()
            self.x_max = self.buf0.max()
            self.y_min = self.buf1.min()
            self.y_max = self.buf1.max()



    def get_raw_data(self):
        x = np.roll(self.buf0, -self.ptr)
        y = np.roll(self.buf1, -self.ptr)
        ttl = np.roll(self.bufz, -self.ptr)
        t = np.roll(self.buft, -self.ptr)
        phi = np.roll(self.bufphi, -self.phi_ptr)
        phi_t = np.roll(self.bufphit, -self.phi_ptr)
        amp = np.roll(self.bufamp, -self.phi_ptr)

        return x,y,ttl,t, phi, phi_t, amp


# def find_trigger_point(data, level=0.0):
#     for i in range(len(data) - 1):
#         if data[i] <= level and data[i + 1] > level:
#             return i
#     return 0  # Default to start if no trigger found
# def filter_data():
#     pass
    # # rolling view
    # view0 = np.roll(buf0, -ptr)
    # view1 = np.roll(buf1, -ptr)
    # viewz = np.roll(bufz, -ptr)
    # viewt = np.roll(buft, -ptr)

    # # Filter the display buffer (zero-phase)
    # if filter_enabled:
    #     view0_filt = filtfilt(filt_b, filt_a, view0)
    #     view1_filt = filtfilt(filt_b, filt_a, view1)
    #     viewz_filt = viewz  # Keep raw z values for triggering and display
    #     viewt_filt = viewt * 1e-6  # Keep raw t values for triggering and display
    #     # viewz_filt = filtfilt(filt_b, filt_a, viewz)
    # else:
    #     view0_filt = view0
    #     view1_filt = view1
    #     viewz_filt = viewz
    #     viewt_filt = viewt * 1e-6


    # # Apply trigger on filtered data
    # if trigger_enabled:
    #     trigger_data = viewz_filt if trigger_channel == 0 else view1_filt
    #     trigger_level = viewz_filt.mean() 
    #     trigger_idx = find_trigger_point(trigger_data, trigger_level)
    #     trigger_idx += trigger_offset
        
    #     # Instead of rolling, slice from trigger point and pad with NaN to avoid wrap-around artifacts
    #     valid_length = len(view0_filt) - trigger_idx
    #     view0_filt_triggered = np.full_like(view0_filt, np.nan, dtype=np.float64)
    #     view1_filt_triggered = np.full_like(view1_filt, np.nan, dtype=np.float64)
    #     view_z_filt_triggered = np.full_like(viewz, np.nan, dtype=np.float64)
    #     view_t_filt_triggered = np.full_like(viewt_filt, np.nan, dtype=np.float64)

    #     view0_filt_triggered[:valid_length] = view0_filt[trigger_idx:]
    #     view1_filt_triggered[:valid_length] = view1_filt[trigger_idx:]
    #     view_z_filt_triggered[:valid_length] = viewz[trigger_idx:]
    #     view_t_filt_triggered[:valid_length] = viewt_filt[trigger_idx:]

    #     view0_filt = view0_filt_triggered
    #     view1_filt = view1_filt_triggered
    #     viewz_filt = view_z_filt_triggered
    #     viewt_filt = view_t_filt_triggered - np.nanmin(view_t_filt_triggered)  # Normalize time to start at zero

    # return view0, view0_filt, view1, view1_filt, viewz, viewz_filt, viewt, viewt_filt
