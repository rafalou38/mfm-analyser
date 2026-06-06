from datetime import datetime
import numpy as np
import subprocess
import os
import select
timestamp = datetime.now().strftime("%Y%m%d_%H:%M:%S")
class LockIn:
    def __init__(self, history):
        self.ptr = 0
        self.phi_ptr = 0

        self.proc = subprocess.Popen(
            ["./serial"], # lecture série temps réel (programme c)
            # ["cat", "./free.bin"], # enregistrement sauvegardé
            stdout=subprocess.PIPE,
            bufsize=0
        )

        self.FRAME_BYTES = 256 * (4 + 4 + 1 + 4)
        self.history_sz = history
        self.phi_history_sz = history * 10

        self.buf0 = np.zeros(self.history_sz, dtype=np.float128)
        self.buf1 = np.zeros(self.history_sz, dtype=np.float128)
        self.bufz = np.zeros(self.history_sz, dtype=np.int8)
        self.buft = np.zeros(self.history_sz, dtype=np.float128)

        self.last_phi = 0.0
        self.phase_acc = 0.0
        self.last_edge_distance = None
        self.bufphi = np.zeros(self.phi_history_sz, dtype=np.float32)
        self.bufamp = np.zeros(self.phi_history_sz, dtype=np.float32)
        self.bufphit = np.zeros(self.phi_history_sz, dtype=np.float32)

        self.buf_phi_simple = np.zeros(self.phi_history_sz, dtype=np.float32)
        self.buf_amp_simple = np.zeros(self.phi_history_sz, dtype=np.float32)
        self.buf_simple_t = np.zeros(self.phi_history_sz, dtype=np.float32)
        self.buf_simple_ptr = 0

        self.x_min = 0.0
        self.x_max = 0.01
        self.y_min = 0.0
        self.y_max = 0.01

        self.ready = False

        self.sos = None
        self.zi_I = None
        self.zi_Q = None

        self.measure_freqs = []
        self.measure_phi = []
        self.measure_amp = []

        self.s = np.zeros(self.history_sz, dtype=np.float32)

        self.peaks = []

        self.i = 0

    # Rattrape le retard
    def clear_serial(self):
        while select.select([self.proc.stdout.fileno()], [], [], 0)[0]:
            os.read(self.proc.stdout.fileno(), 65536)

    # Lecture des données série
    def read_serial(self, phase=True):
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

        # Gestion du buffer circulaire
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

        if phase:
            self.handle_phase_simple(self.buf0, self.bufz, self.buft)

        self.update_bounds()
    

    def reset_bounds(self):
        self.x_min = 0.0
        self.x_max = 0.01
        self.y_min = 0.0
        self.y_max = 0.01

    def update_bounds(self):
        self.x_min = min(self.buf0.min(), self.x_min)
        self.x_max = max(self.buf0.max(), self.x_max)
        self.y_min = min(self.buf1.min(), self.y_min)
        self.y_max = max(self.buf1.max(), self.y_max)

        if not np.isfinite([self.x_min, self.x_max, self.y_min, self.y_max]).all() or max(abs(self.x_min), abs(self.x_max), abs(self.y_min), abs(self.y_max)) > 1e6:
            self.x_min = self.buf0.min()
            self.x_max = self.buf0.max()
            self.y_min = self.buf1.min()
            self.y_max = self.buf1.max()

    # Restructuration du buffer circulaire
    def get_raw_data(self):
        x = np.roll(self.buf0, -self.ptr)
        y = np.roll(self.buf1, -self.ptr)
        ttl = np.roll(self.bufz, -self.ptr)
        t = np.roll(self.buft, -self.ptr)
        phi = np.roll(self.bufphi, -self.phi_ptr)
        phi_t = np.roll(self.bufphit, -self.phi_ptr)
        amp = np.roll(self.bufamp, -self.phi_ptr)
        s = np.roll(self.s, -self.phi_ptr)

        phi_simple = np.roll(self.buf_phi_simple, -self.buf_simple_ptr)
        amp_simple = np.roll(self.buf_amp_simple, -self.buf_simple_ptr)
        phi_simple_t = np.roll(self.buf_simple_t, -self.buf_simple_ptr)

        return x,y,ttl,t, phi, phi_t, amp, s, phi_simple, phi_simple_t, amp_simple

    def handle_phase_simple(self, x_values, z_values, t_values):
        avg = np.mean(x_values)

        x_pad = np.pad(x_values, (4, 5), mode='edge')

        # Détection des pics sur le signal
        n = len(x_values)
        back5  = np.array([np.mean(x_pad[i:i+5])   for i in range(n)])
        front5 = np.array([np.mean(x_pad[i+5:i+10]) for i in range(n)])
        edges = np.where((back5[:-1] < avg) & (front5[1:] >= avg))[0]

        # Detection des créneaux
        filtered_edges = []
        for i in range(1, len(edges)):
            # 1/100 pour 100 Hz
            if t_values[edges[i]] - t_values[edges[i-1]] > 1/100:
                filtered_edges.append(edges[i-1])
        filtered_edges.append(edges[-1])

        self.peaks = [t_values[i] for i in filtered_edges]

        # Temps d’échantillonnage
        # Ts = abs(np.median(np.diff(t_values)))
        # fs = 1 / Ts
        # print(fs)

        edges_ref = np.where((z_values[:-1] == 0) & (z_values[1:] == 1))[0]
        d = np.diff(t_values[edges_ref])
        t0 = np.median(d)
        # print("\n Période médiane: ", 1/t0, " Hz ")

        phases = []

        # TTL suivant
        # for i in filtered_edges:
        #     # Find next reference edge
        #     next_edge = edges_ref[edges_ref > i].min() if edges_ref[edges_ref > i].size > 0 else None
        #     if next_edge is not None:
        #         phases.append((t_values[i] - t_values[next_edge]) * 2 * np.pi / t0)

        # TTL le plus proche
        for i in filtered_edges:
            diffs = np.abs(edges_ref - i)
            nearest_edge = edges_ref[np.argmin(diffs)]
            phases.append((t_values[i] - t_values[nearest_edge]) * 2 * np.pi / t0)


        # Stockage des valeurs
        self.buf_simple_t[self.buf_simple_ptr] = (np.max(t_values) + np.min(t_values))/2
        self.buf_phi_simple[self.buf_simple_ptr] = np.median(phases)
        self.buf_amp_simple[self.buf_simple_ptr] = np.max(x_pad) - np.min(x_pad)

        self.buf_phi_simple = np.unwrap(self.buf_phi_simple)
        # Donnés enregistrées en continu
        with open(f"phase-{timestamp}.csv", "a") as f:
            f.write(f"{self.buf_simple_t[self.buf_simple_ptr]},{1/t0},{np.degrees(self.buf_phi_simple[self.buf_simple_ptr])},{np.max(x_pad) - np.min(x_pad)},{np.std(phases)}\n")

        self.buf_simple_ptr += 1
        if self.buf_simple_ptr >= self.phi_history_sz:
            self.buf_simple_ptr = 0


    def get_trigger(self):
        t = np.roll(self.buft, -self.ptr)
        ttl = np.roll(self.bufz, -self.ptr)

        edges = np.where(
            (ttl[:-1] == 0) &
            (ttl[1:] == 1)
        )[0]

        if len(edges) == 0:
            return t[0]

        return t[edges[0] + 1]

    def reset_phase(self):
        self.last_phi = 0.0