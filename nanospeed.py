import serial
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import time
import subprocess
from scipy.signal import butter, filtfilt

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
FRAME_BYTES = FRAME_N * (4 + 4 + 1)
proc = subprocess.Popen(
    ["./serial"],
    stdout=subprocess.PIPE,
    bufsize=0
)

ptr = 0
buf0 = np.zeros(HISTORY, dtype=np.float64)
buf1 = np.zeros(HISTORY, dtype=np.float64)
bufz = np.zeros(HISTORY, dtype=np.int8)

# Trigger settings
filter_enabled = False
trigger_enabled = True
trigger_channel = 0  # 0 for buf0, 1 for buf1
trigger_level = 0.1
trigger_offset = 0  # Offset from trigger point

# ########
# FILTRAGE
# ########

filt_b, filt_a = butter(3, 0.15, btype='low')

################################
################################
### GRAPHES
################################
################################

# ############
# Signaux
# ############

app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="NanoSpeed")




x_min = 0.0
x_max = 0.01
y_min = 0.0
y_max = 0.01


class MainPlot:
    def __init__(self, win):
        self.plot = win.addPlot(title="Signals")
        self.plot.setYRange(0, 1)
        self.plot.showGrid(x=True, y=True, alpha=0.3)  # Add grid to main plot
        self.curve0 = self.plot.plot(pen='y', name="A0")
        self.curve1 = self.plot.plot(pen='c', name="A1")
        self.curvez = self.plot.plot(pen='m', name="Az")

class FFTView:
    def __init__(self, win):
        self.plot = win.addPlot(title="FFT")
        self.plot.setYRange(0, 1)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen='w')

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
    def update(self):
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

cursor_view = CursorView(win)
main_plot = MainPlot(win)

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

    # Filter the display buffer (zero-phase)
    if filter_enabled:
        view0_filt = filtfilt(filt_b, filt_a, view0)
        view1_filt = filtfilt(filt_b, filt_a, view1)
        viewz_filt = viewz  # Keep raw z values for triggering and display
        # viewz_filt = filtfilt(filt_b, filt_a, viewz)
    else:
        view0_filt = view0
        view1_filt = view1
        viewz_filt = viewz

    # Apply trigger on filtered data
    if trigger_enabled:
        trigger_data = viewz_filt if trigger_channel == 0 else view1_filt
        trigger_level = viewz_filt.mean() 
        trigger_idx = find_trigger_point(trigger_data, trigger_level)
        trigger_idx += trigger_offset
        
        # Instead of rolling, slice from trigger point and pad with NaN to avoid wrap-around artifacts
        valid_length = len(view0_filt) - trigger_idx
        view0_filt_triggered = np.full_like(view0_filt, np.nan)
        view1_filt_triggered = np.full_like(view1_filt, np.nan)
        view_z_filt_triggered = np.full_like(viewz, np.nan)


        
        view0_filt_triggered[:valid_length] = view0_filt[trigger_idx:]
        view1_filt_triggered[:valid_length] = view1_filt[trigger_idx:]
        view_z_filt_triggered[:valid_length] = viewz[trigger_idx:]

        view0_filt = view0_filt_triggered
        view1_filt = view1_filt_triggered
        viewz_filt = view_z_filt_triggered

    return view0, view0_filt, view1, view1_filt, viewz, viewz_filt

def draw():
    view0_raw, view0_filt, view1_raw, view1_filt, viewz_raw, viewz_filt = filter_data()

    main_plot.curve0.setData(view0_filt)
    main_plot.curve1.setData(view1_filt)
    main_plot.curvez.setData(viewz_filt * max(x_max, y_max))

    # Safe range setting with bounds checking
    y_range_min = float(min(y_min, x_min) * 2)
    y_range_max = float(max(x_max, y_max) * 5)
    if np.isfinite([y_range_min, y_range_max]).all() and abs(y_range_max - y_range_min) < 1e6:
        main_plot.plot.setYRange(y_range_min, y_range_max)

    
    cursor_view.update()
    cursor_view.set_data(
        view0_raw[-256:],
        view1_raw[-256:]
    )

def read_serial():
    global ptr

    data = proc.stdout.read(FRAME_BYTES)
    if len(data) != FRAME_BYTES:
        print("End of stream")
        return
    dtype = np.dtype([
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.int8),
    ])
    samples = np.frombuffer(data, dtype=dtype)

    x_values = samples['x']
    y_values = samples['y']
    z_values = samples['z']

    # Store RAW data
    n = len(x_values)
    if ptr + n < HISTORY:
        buf0[ptr:ptr+n] = x_values
        buf1[ptr:ptr+n] = y_values
        bufz[ptr:ptr+n] = z_values
    else:
        k = HISTORY - ptr
        buf0[ptr:] = x_values[:k]
        buf1[ptr:] = y_values[:k]
        bufz[ptr:] = z_values[:k]

        buf0[:n-k] = x_values[k:]
        buf1[:n-k] = y_values[k:]
        bufz[:n-k] = z_values[k:]

    ptr = (ptr + n) % HISTORY


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

    draw()



timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

app.exec()