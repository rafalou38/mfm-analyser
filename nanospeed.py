import serial
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import time
import subprocess
from scipy.signal import butter, filtfilt

HISTORY = 4000          # visible history

################################
################################
### SIGNALS
################################
################################


# ############
# Data
# ############
FRAME_FLOATS = 512
FRAME_BYTES = FRAME_FLOATS * 4
proc = subprocess.Popen(
    ["./serial"],
    stdout=subprocess.PIPE,
    bufsize=0
)

ptr = 0
buf0 = np.zeros(HISTORY, dtype=np.float64)

buf1 = np.zeros(HISTORY, dtype=np.float64)

# ########
# FILTRAGE
# ########

filt_b, filt_a = butter(3, 0.05, btype='low')

################################
################################
### GRAPHES
################################
################################


app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Nano Oscilloscope")
# ############
# Signaux
# ############
plot = win.addPlot(title="Signals")
plot.setYRange(0, 1)
curve0 = plot.plot(pen='y', name="A0")
curve1 = plot.plot(pen='c', name="A1")

overlay = pg.ViewBox()
plot.scene().addItem(overlay)

overlay.setXRange(-1,1)
overlay.setYRange(-1,1)
overlay.setAspectLocked(True)

def update_overlay_geometry():
    rect = plot.getViewBox().sceneBoundingRect()

    w = rect.width() * 0.35
    h = rect.height() * 0.35

    overlay.setGeometry(
        rect.x() + rect.width() - w,
        rect.y(),
        w,
        h
    )

plot.getViewBox().sigResized.connect(update_overlay_geometry)
update_overlay_geometry()

# ############
# Pointeur
# ############
vline = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('w', width=1))
hline = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('w', width=1))

circle = QtWidgets.QGraphicsEllipseItem()
circle.setPen(pg.mkPen('w', width=1))
scale_text = pg.TextItem(anchor=(1,1), color='w')
overlay.addItem(scale_text)

scatter = pg.ScatterPlotItem(size=4, brush='r', pen=None)

overlay.addItem(scatter)
overlay.addItem(vline)
overlay.addItem(hline)
overlay.addItem(circle)

x_min = 0.0
x_max = 0.01
y_min = 0.0
y_max = 0.01

alpha = np.linspace(255, 0, 256).astype(np.uint8)
colors = np.zeros((256, 4), dtype=np.uint8)
colors[:,0] = 255
colors[:,3] = alpha

def update():
    global ptr, x_min, x_max, y_min, y_max

    data = proc.stdout.read(FRAME_BYTES)
    if len(data) != FRAME_BYTES:
        print("End of stream")
        return
    
    samples = np.frombuffer(data, dtype=np.float32)

    x_values = samples[0::2]
    y_values = samples[1::2]

    x_values = filtfilt(filt_b, filt_a, x_values)
    y_values = filtfilt(filt_b, filt_a, y_values)

    n = len(x_values)
    if ptr + n < HISTORY:
        buf0[ptr:ptr+n] = x_values
        buf1[ptr:ptr+n] = y_values
    else:
        k = HISTORY - ptr
        buf0[ptr:] = x_values[:k]
        buf1[ptr:] = y_values[:k]
        buf0[:n-k] = x_values[k:]
        buf1[:n-k] = y_values[k:]
    ptr = (ptr + n) % HISTORY

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

    # rolling view
    view0 = np.roll(buf0, -ptr)
    view1 = np.roll(buf1, -ptr)
    curve0.setData(view0)
    curve1.setData(view1)
    
    # Safe range setting with bounds checking
    y_range_min = float(min(y_min, x_min) * 2)
    y_range_max = float(max(x_max, y_max) * 5)
    if np.isfinite([y_range_min, y_range_max]).all() and abs(y_range_max - y_range_min) < 1e6:
        plot.setYRange(y_range_min, y_range_max)

    # Update overlay pointer
    r = float(max(abs(x_min), abs(x_max), abs(y_min), abs(y_max)))
    if not np.isfinite(r) or r > 10 or r < 1e-10:
        return
    
    scale_text.setText(f"±{r:.3f}")
    scale_text.setPos(r, r)

    overlay.setXRange(float(-r), float(r))
    overlay.setYRange(float(-r), float(r))

    circle.setRect(float(-r), float(-r), float(2*r), float(2*r))
    
    scatter.setData(x_values, y_values, brush=colors)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

app.exec()