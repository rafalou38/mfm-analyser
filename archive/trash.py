# Mostly unusable ai hallucinated trash with some tweaks (ditched in favor of handmade LockIn.py)

class Fah:
    def compute_phase_from_buffers(self):
        """
        Compute average phase difference between:
            - analog signal buf0
            - TTL reference bufz

        using lock-in style IQ demodulation
        on the CURRENT FULL BUFFERS.

        Returns:
            freq_hz
            avg_phase_rad
            avg_amplitude
        """

        # -------------------------------------------------
        # GET CONTIGUOUS BUFFER VIEWS
        # -------------------------------------------------

        x = np.roll(self.buf0, -self.ptr)
        ttl = np.roll(self.bufz, -self.ptr)
        t = np.roll(self.buft, -self.ptr)

        # -------------------------------------------------
        # SAMPLING FREQUENCY
        # -------------------------------------------------

        dt = np.diff(t)

        dt = dt[dt > 0]

        if len(dt) == 0:
            return None

        Ts = np.median(dt)

        fs = 1.0 / Ts

        # -------------------------------------------------
        # TTL EDGE DETECTION
        # -------------------------------------------------

        edges = np.where(
            (ttl[:-1] == 0) &
            (ttl[1:] == 1)
        )[0]

        if len(edges) < 2:
            print("Not enough TTL edges")
            return None

        # -------------------------------------------------
        # FREQUENCY ESTIMATION
        # -------------------------------------------------

        periods = np.diff(edges)

        mean_period = np.median(periods)

        freq_hz = fs / mean_period

        # -------------------------------------------------
        # PHASE RECONSTRUCTION
        # -------------------------------------------------

        phase = np.zeros_like(x)

        for k in range(len(edges) - 1):

            i0 = edges[k]
            i1 = edges[k + 1]

            L = i1 - i0

            if L <= 0:
                continue

            phase[i0:i1] = np.linspace(
                0,
                2 * np.pi,
                L,
                endpoint=False
            )

        # -------------------------------------------------
        # IQ DEMODULATION
        # -------------------------------------------------

        s = np.sin(phase)
        c = np.cos(phase)

        I_raw = x * c
        Q_raw = x * s

        # -------------------------------------------------
        # LOW PASS FILTER
        # -------------------------------------------------

        sos = butter(
            3,
            5,
            fs=fs,
            output='sos'
        )

        I = sosfilt(sos, I_raw)
        Q = sosfilt(sos, Q_raw)

        # -------------------------------------------------
        # COMPLEX LOCK-IN OUTPUT
        # -------------------------------------------------

        Z = I + 1j * Q

        # discard filter transient
        Z = Z[len(Z)//4:]

        # -------------------------------------------------
        # COMPLEX AVERAGING
        # -------------------------------------------------

        avg_Z = np.mean(Z)

        avg_amp = np.abs(avg_Z)

        avg_phase = np.angle(avg_Z)

        print(
            f"f = {freq_hz:.3f} Hz | "
            f"phase = {avg_phase:.4f} rad | "
            f"phase = {np.degrees(avg_phase):.4f} deg | "
            f"amp = {avg_amp:.6f}"
        )

        return freq_hz, avg_phase, avg_amp


        if red > 1:
            m = len(x_values) // red

            # Trim to multiple of red
            x_values = x_values[:m * red]
            y_values = y_values[:m * red]
            z_values = z_values[:m * red]
            t_values = t_values[:m * red]

            # Average groups
            x_values = x_values.reshape(m, red).mean(axis=1)
            y_values = y_values.reshape(m, red).mean(axis=1)

            # z: keep center sample
            z_values = z_values.reshape(m, red)[:, red // 2]

            # t: use center point time
            t_values = t_values.reshape(m, red)[:, red // 2]



    def save_average_point(
        self,
        avg_time=2.0,
        settle_time=1.0,
    ):
        """
        Measure average lock-in response and save one point.
        """

        print("Settling...")

        # -------------------------------------------------
        # SETTLING
        # -------------------------------------------------

        t0 = time.time()

        while time.time() - t0 < settle_time:
            self.read_serial()

        print("Averaging...")

        # -------------------------------------------------
        # ACQUISITION
        # -------------------------------------------------

        Z_samples = []
        freq_samples = []

        t0 = time.time()

        while time.time() - t0 < avg_time:

            self.read_serial()

            if not hasattr(self, "current_Z"):
                continue

            # Take newest chunk only
            Z = self.current_Z[-256:]

            Z_samples.append(Z)

            if hasattr(self, "current_freq"):
                freq_samples.append(self.current_freq)

        # -------------------------------------------------
        # CONCATENATE
        # -------------------------------------------------

        if len(Z_samples) == 0:
            print("No samples acquired")
            return

        Z_samples = np.concatenate(Z_samples)

        # -------------------------------------------------
        # COMPLEX AVERAGING
        # -------------------------------------------------

        avg_Z = np.mean(Z_samples)

        avg_amp = np.abs(avg_Z)

        avg_phi = np.angle(avg_Z)

        avg_freq = np.mean(freq_samples)

        # -------------------------------------------------
        # STORE
        # -------------------------------------------------

        self.measure_freqs.append(avg_freq)
        self.measure_phi.append(avg_phi)
        self.measure_amp.append(avg_amp)

        print(
            f"Saved point: "
            f"f={avg_freq:.3f} Hz  "
            f"phi={avg_phi:.3f} rad  "
            f"A={avg_amp:.6f}"
        )

        with open(f"lockin_data{os.getpid()}.csv", "a") as f:
            f.write(f"{avg_freq:.3f},{avg_phi:.3f},{avg_amp:.6f}\n")



    def compute_phase_offset_simple(self):
        """
        Estimate phase offset between:
            - analog signal buf0
            - TTL reference bufz

        using peak timing only.

        Returns:
            phase_rad
        """

        # -------------------------------------------------
        # GET CONTIGUOUS BUFFERS
        # -------------------------------------------------

        x = np.roll(self.buf0, -self.ptr)
        ttl = np.roll(self.bufz, -self.ptr)
        t = np.roll(self.buft, -self.ptr)

        # -------------------------------------------------
        # TTL RISING EDGES
        # -------------------------------------------------


        edges = np.where(
            (ttl[:-1] == 0) &
            (ttl[1:] == 1)
        )[0]

        if len(edges) < 2:
            print("Not enough TTL edges")
            return None

        # -------------------------------------------------
        # SIGNAL PEAKS
        # -------------------------------------------------

        filt_b, filt_a = butter(3, 0.15, btype='low')
        self.x_filt = filtfilt(filt_b, filt_a, x)
        peaks, _ = find_peaks(
            self.x_filt,
            prominence=np.std(self.x_filt) * 0.5
        )

        self.peaks = [
            (self.buft[p], x[p])
            for p in peaks
        ]

        if len(peaks) == 0:
            print("No peaks found")
            return None

        # -------------------------------------------------
        # MATCH EACH TTL EDGE TO CLOSEST SIGNAL PEAK
        # -------------------------------------------------

        phase_offsets = []

        for e in edges[:-1]:

            # next ttl edge defines one period
            next_edge = edges[np.searchsorted(edges, e) + 1]

            period_samples = next_edge - e

            # peaks inside this cycle
            cycle_peaks = peaks[
                (peaks >= e) &
                (peaks < next_edge)
            ]

            if len(cycle_peaks) == 0:
                continue

            # take first peak
            p = cycle_peaks[0]

            # sample offset
            dt_samples = p - e

            # convert to phase
            phi = 2 * np.pi * dt_samples / period_samples

            phase_offsets.append(phi)

        if len(phase_offsets) == 0:
            print("No valid phase measurements")
            return None


        print("Avg phase offset (simple): ", np.degrees(np.mean(phase_offsets)), " deg")
        print("Std phase offset (simple): ", np.degrees(np.std(phase_offsets)), " deg\n")