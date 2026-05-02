import serial
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import time
import subprocess
from scipy.signal import butter, filtfilt, sosfilt, sosfilt_zi

HISTORY = 10000          # visible history

################################
################################
### SIGNALS
################################
################################


# ############
# Data
# ############
FRAME_N = 256
FRAME_BYTES = FRAME_N * (4 + 4 + 1 + 4)
proc = subprocess.Popen(
    ["./serial"],
    stdout=subprocess.PIPE,
    bufsize=0
)

ptr = 0
buf0 = np.zeros(HISTORY, dtype=np.float64)
buf1 = np.zeros(HISTORY, dtype=np.float64)
bufz = np.zeros(HISTORY, dtype=np.int8)
buft = np.zeros(HISTORY, dtype=np.float64)

# Trigger settings
filter_enabled = False
filt_b, filt_a = butter(3, 0.15, btype='low')


trigger_enabled = False
fft_enabled = False
trigger_channel = 0  # 0 for buf0, 1 for buf1
trigger_level = 0.1
trigger_offset = 0  # Offset from trigger point

ready = False

x_min = 0.0
x_max = 0.01
y_min = 0.0
y_max = 0.01


class MainPlot:
    def __init__(self, win):
        self.plot = win.addPlot(title="Signals")
        self.plot.enableAutoRange('x', True)
        self.plot.setYRange(0, 1)
        self.plot.showGrid(x=True, y=True, alpha=0.3)  # Add grid to main plot
        self.curve0 = self.plot.plot(pen='y', name="A0")
        self.curve1 = self.plot.plot(pen='c', name="A1")
        self.curvez = self.plot.plot(pen='m', name="Az")
        self.curvec = self.plot.plot(pen='m', name="Az")

        self.peak_items = []

        self.sos = None

        self.zi_I = None
        self.zi_Q = None

    def draw(self, t, x, y, ttl):
        self.curve0.setData(t,x)
        self.curve1.setData(t, y)

        # Find nan range
        nan_indices = np.where(np.isnan(x))[0]
        if len(nan_indices) > 0:
            print(nan_indices[0], nan_indices[-1])

        Ts = abs(np.median(np.diff(buft))) * 1e-6
        fs = 1 / Ts

        if self.sos is None:
            self.sos = butter(3, 5, fs=fs, output='sos')
            n_sections = self.sos.shape[0]

            self.zi_I = np.zeros((n_sections, 2))
            self.zi_Q = np.zeros((n_sections, 2))
            print(fs)

        edges = np.where((ttl[:-1] == 0) & (ttl[1:] == 1))[0]

        # Soft Sync
        # T0 = abs(np.median(np.diff(t[edges])))
        # f0 = 1 / T0
        # s = np.sin(2 * np.pi * f0 * t)

        # Hard sync
        phase = np.zeros_like(t)
        for k in range(len(edges)-1):
            i0 = edges[k]
            i1 = edges[k+1]

            phase[i0:i1] = np.linspace(0, 2*np.pi, i1-i0, endpoint=False)
        s = np.sin(phase)
        c = np.cos(phase)

        # self.curvez.setData(t, c * max(x_max, y_max))

        I_raw = x * c
        Q_raw = x * s
        # sos = butter(3, 5, fs=fs, output='sos')
        # I = sosfilt(sos, I_raw)
        # Q = sosfilt(sos, Q_raw)

        I, self.zi_I = sosfilt(self.sos, I_raw, zi=self.zi_I)
        Q, self.zi_Q = sosfilt(self.sos, Q_raw, zi=self.zi_Q)


        phi = np.arctan2(Q, I)
        phi = np.unwrap(phi)

        # A = np.sqrt(I**2 + Q**2)

        self.curvez.setData(t, phi * max(x_max, y_max))

        # Safe range setting with bounds checking
        y_range_min = float(min(y_min, x_min) * 2)
        y_range_max = float(max(x_max, y_max) * 5)
        if np.isfinite([y_range_min, y_range_max]).all() and abs(y_range_max - y_range_min) < 1e6:
            self.plot.setYRange(y_range_min, y_range_max)


class PhaseView:
    def __init__(self, win):
        self.plot = win.addPlot(title="Phase", row=1, col=2)
        self.plot.enableAutoRange(True)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen='w')

    def update(self):
        # np.corel
        peaks, _ = find_peaks(buf0, height=0.2, prominence=0.05, distance=5)

class FFTView:
    def __init__(self, win):
        self.plot = win.addPlot(title="FFT", row=1, col=0)
        self.plot.enableAutoRange(True)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen='w')
        self.i = 0
        self.labels = []

    # def update():

    def setData(self, e):
        # if np.isnan(data).any(): return

        data = buf0
        data = np.nan_to_num(data)

        sp = rfft(data)
        dt = abs(np.median(np.diff(buft))) * 1e-6
        self.i+=1
        if self.i % 100 ==0:
            print("Sampling: ", round(1/dt), "Hz")
        freqs = rfftfreq(len(data), d=dt)
        magnitude = np.abs(sp)
        magnitude /= np.max(magnitude)

        f_view = freqs[freqs<150]
        m_view = magnitude[freqs<150]

        peaks, _ = find_peaks(magnitude, height=0.2,
            prominence=0.05,
            distance=5)
        
        # Remove old labels
        for label in self.labels:
            self.plot.removeItem(label)

        self.labels.clear()
        for p in peaks:
            if p >= len(f_view) or p >= len(m_view):
                continue
            f = f_view[p]
            a = m_view[p]

            text = pg.TextItem(
                text=f"{f:.1f} Hz",
                color='y',
                anchor=(0.5, 1.0)
            )

            text.setPos(f, a)

            self.plot.addItem(text)

            self.labels.append(text)

        self.curve.setData(f_view, m_view)

class CursorView:
    def __init__(self, win):

        overlay = pg.ViewBox()
        overlay.setZValue(1000)
        overlay.setXRange(-1,1)
        overlay.setYRange(-1,1)
        overlay.setAspectLocked(True)
        overlay.setBackgroundColor((0, 0, 0, 200))
        win.scene().addItem(overlay)

        self._overlay = overlay
        # Deco
        self.radial_grid_circles = []
        for i in range(1, 5):  # 4 concentric circles at 0.25, 0.5, 0.75, 1.0
            circle_item = QtWidgets.QGraphicsEllipseItem()
            circle_item.setPen(pg.mkPen((100, 100, 100), width=1, style=QtCore.Qt.PenStyle.DashLine))
            self.radial_grid_circles.append(circle_item)
            overlay.addItem(circle_item)

        radial_grid_lines = []
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:  # 8 radial lines
            line = pg.InfiniteLine(angle=angle, pen=pg.mkPen((100, 100, 100), width=1, style=QtCore.Qt.PenStyle.DashLine))
            radial_grid_lines.append(line)
            overlay.addItem(line)

        vline = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('w', width=1))
        hline = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('w', width=1))

        self.circle = QtWidgets.QGraphicsEllipseItem()
        self.circle.setPen(pg.mkPen('w', width=2))
        self.scale_text = pg.TextItem(anchor=(1,1), color='w')
        overlay.addItem(self.scale_text)

        # Graph
        self.scatter = pg.ScatterPlotItem(size=4, brush='r', pen=None)

        overlay.addItem(self.scatter)
        overlay.addItem(vline)
        overlay.addItem(hline)
        overlay.addItem(self.circle)


        alpha = np.linspace(255, 0, 256).astype(np.uint8)
        self.colors = np.zeros((256, 4), dtype=np.uint8)
        self.colors[:,0] = 255
        self.colors[:,3] = alpha
    def draw(self):
        # Update overlay pointer
        r = float(max(abs(x_min), abs(x_max), abs(y_min), abs(y_max)))
        if not np.isfinite(r) or r > 10 or r < 1e-10:
            return
        
        self.scale_text.setText(f"±{r:.3f}")
        self.scale_text.setPos(r, r)

        self._overlay.setXRange(float(-r), float(r))
        self._overlay.setYRange(float(-r), float(r))

        self.circle.setRect(float(-r), float(-r), float(2*r), float(2*r))
        
        # Update radial grid circles
        for i, circle_item in enumerate(self.radial_grid_circles):
            radius = r * (i + 1) / 4  # Divide into 4 sections
            circle_item.setRect(float(-radius), float(-radius), float(2*radius), float(2*radius))
    
    def update_geometry(self, plot):
        rect = plot.getViewBox().sceneBoundingRect()

        w = rect.width() * 0.35
        h = rect.height() * 0.35

        self._overlay.setGeometry(
            rect.x() + rect.width() - w,
            rect.y(),
            w,
            h
        )

    def set_data(self, x, y):
        self.scatter.setData(x, y, brush=self.colors)


app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="NanoSpeed")

cursor_view = CursorView(win)
main_plot = MainPlot(win)
fft_view = FFTView(win)

main_plot.plot.getViewBox().sigResized.connect(lambda: cursor_view.update_geometry(main_plot.plot))
cursor_view.update_geometry(main_plot.plot)

def find_trigger_point(data, level=0.0):
    for i in range(len(data) - 1):
        if data[i] <= level and data[i + 1] > level:
            return i
    return 0  # Default to start if no trigger found

def filter_data():
    # rolling view
    view0 = np.roll(buf0, -ptr)
    view1 = np.roll(buf1, -ptr)
    viewz = np.roll(bufz, -ptr)
    viewt = np.roll(buft, -ptr)

    # Filter the display buffer (zero-phase)
    if filter_enabled:
        view0_filt = filtfilt(filt_b, filt_a, view0)
        view1_filt = filtfilt(filt_b, filt_a, view1)
        viewz_filt = viewz  # Keep raw z values for triggering and display
        viewt_filt = viewt * 1e-6  # Keep raw t values for triggering and display
        # viewz_filt = filtfilt(filt_b, filt_a, viewz)
    else:
        view0_filt = view0
        view1_filt = view1
        viewz_filt = viewz
        viewt_filt = viewt * 1e-6


    # Apply trigger on filtered data
    if trigger_enabled:
        trigger_data = viewz_filt if trigger_channel == 0 else view1_filt
        trigger_level = viewz_filt.mean() 
        trigger_idx = find_trigger_point(trigger_data, trigger_level)
        trigger_idx += trigger_offset
        
        # Instead of rolling, slice from trigger point and pad with NaN to avoid wrap-around artifacts
        valid_length = len(view0_filt) - trigger_idx
        view0_filt_triggered = np.full_like(view0_filt, np.nan, dtype=np.float64)
        view1_filt_triggered = np.full_like(view1_filt, np.nan, dtype=np.float64)
        view_z_filt_triggered = np.full_like(viewz, np.nan, dtype=np.float64)
        view_t_filt_triggered = np.full_like(viewt_filt, np.nan, dtype=np.float64)

        view0_filt_triggered[:valid_length] = view0_filt[trigger_idx:]
        view1_filt_triggered[:valid_length] = view1_filt[trigger_idx:]
        view_z_filt_triggered[:valid_length] = viewz[trigger_idx:]
        view_t_filt_triggered[:valid_length] = viewt_filt[trigger_idx:]

        view0_filt = view0_filt_triggered
        view1_filt = view1_filt_triggered
        viewz_filt = view_z_filt_triggered
        viewt_filt = view_t_filt_triggered - np.nanmin(view_t_filt_triggered)  # Normalize time to start at zero

    return view0, view0_filt, view1, view1_filt, viewz, viewz_filt, viewt, viewt_filt

    
def read_serial():
    global ptr, ready

    data = proc.stdout.read(FRAME_BYTES)
    if len(data) != FRAME_BYTES:
        print("End of stream")
        return
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
    t_values = samples['t']
    # Store RAW data
    n = len(x_values)
    if ptr + n < HISTORY:
        buf0[ptr:ptr+n] = x_values
        buf1[ptr:ptr+n] = y_values
        bufz[ptr:ptr+n] = z_values
        buft[ptr:ptr+n] = t_values
    else:
        ready = True
        k = HISTORY - ptr
        buf0[ptr:] = x_values[:k]
        buf1[ptr:] = y_values[:k]
        bufz[ptr:] = z_values[:k]
        buft[ptr:] = t_values[:k]

        buf0[:n-k] = x_values[k:]
        buf1[:n-k] = y_values[k:]
        bufz[:n-k] = z_values[k:]
        buft[:n-k] = t_values[k:]

    ptr = (ptr + n) % HISTORY

def draw():
    view0_raw, view0_filt, view1_raw, view1_filt, viewz_raw, viewz_filt, viewt_raw, viewt_filt = filter_data()

    # if fft_enabled:
    #     fft_view.setData(view0_raw)
    fft_view.setData(view0_raw)

    main_plot.draw(viewt_raw, view0_raw, view1_raw, viewz_raw)

    cursor_view.set_data(
        view0_raw[-256:],
        view1_raw[-256:],
    )
    cursor_view.draw()

def update():
    global x_min, x_max, y_min, y_max

    read_serial()
    
    x_min = min(buf0.min(), x_min)
    x_max = max(buf0.max(), x_max)
    y_min = min(buf1.min(), y_min)
    y_max = max(buf1.max(), y_max)

    # Reset if values become too large
    if not np.isfinite([x_min, x_max, y_min, y_max]).all() or max(abs(x_min), abs(x_max), abs(y_min), abs(y_max)) > 1e6:
        x_min = buf0.min()
        x_max = buf0.max()
        y_min = buf1.min()
        y_max = buf1.max()

    if ready:
        draw()



timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)


class KeyHandler(QtCore.QObject):
    def eventFilter(self, obj, event):
        global trigger_enabled, filter_enabled
        if event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() == QtCore.Qt.Key.Key_T:
                trigger_enabled = not trigger_enabled
                return True
            if event.key() == QtCore.Qt.Key.Key_F:
                filter_enabled = not filter_enabled
                return True
        return super().eventFilter(obj, event)

key_handler = KeyHandler()
app.installEventFilter(key_handler)

app.exec()