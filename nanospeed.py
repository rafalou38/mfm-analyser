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

# Trigger settings
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


app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Nano Oscilloscope")
# ############
# Signaux
# ############
plot = win.addPlot(title="Signals")
plot.setYRange(0, 1)
plot.showGrid(x=True, y=True, alpha=0.3)  # Add grid to main plot
curve0 = plot.plot(pen='y', name="A0")
curve1 = plot.plot(pen='c', name="A1")

overlay = pg.ViewBox()
plot.scene().addItem(overlay)
overlay.setZValue(1000)

overlay.setXRange(-1,1)
overlay.setYRange(-1,1)
overlay.setAspectLocked(True)
overlay.setBackgroundColor((0, 0, 0, 200))

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
# Create radial grid for overlay
radial_grid_circles = []
for i in range(1, 5):  # 4 concentric circles at 0.25, 0.5, 0.75, 1.0
    circle_item = QtWidgets.QGraphicsEllipseItem()
    circle_item.setPen(pg.mkPen((100, 100, 100), width=1, style=QtCore.Qt.PenStyle.DashLine))
    radial_grid_circles.append(circle_item)
    overlay.addItem(circle_item)

radial_grid_lines = []
for angle in [0, 45, 90, 135, 180, 225, 270, 315]:  # 8 radial lines
    line = pg.InfiniteLine(angle=angle, pen=pg.mkPen((100, 100, 100), width=1, style=QtCore.Qt.PenStyle.DashLine))
    radial_grid_lines.append(line)
    overlay.addItem(line)

vline = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('w', width=1))
hline = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('w', width=1))

circle = QtWidgets.QGraphicsEllipseItem()
circle.setPen(pg.mkPen('w', width=2))
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

def find_trigger_point(data, level=0.0):
    for i in range(len(data) - 1):
        if data[i] <= level and data[i + 1] > level:
            return i
    return 0  # Default to start if no trigger found

def update():
    global ptr, x_min, x_max, y_min, y_max

    data = proc.stdout.read(FRAME_BYTES)
    if len(data) != FRAME_BYTES:
        print("End of stream")
        return
    
    samples = np.frombuffer(data, dtype=np.float32)

    x_values = samples[0::2]
    y_values = samples[1::2]

    # Store RAW data
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
    
    # Filter the display buffer (zero-phase)
    view0_filt = filtfilt(filt_b, filt_a, view0)
    view1_filt = filtfilt(filt_b, filt_a, view1)
    
    # Apply trigger on filtered data
    if trigger_enabled:
        trigger_data = view0_filt if trigger_channel == 0 else view1_filt
        trigger_level = view0_filt.mean() 
        trigger_idx = find_trigger_point(trigger_data, trigger_level)
        trigger_idx += trigger_offset
        
        # Instead of rolling, slice from trigger point and pad with NaN to avoid wrap-around artifacts
        valid_length = len(view0_filt) - trigger_idx
        view0_filt_triggered = np.full_like(view0_filt, np.nan)
        view1_filt_triggered = np.full_like(view1_filt, np.nan)
        
        view0_filt_triggered[:valid_length] = view0_filt[trigger_idx:]
        view1_filt_triggered[:valid_length] = view1_filt[trigger_idx:]
        
        view0_filt = view0_filt_triggered
        view1_filt = view1_filt_triggered
    
    curve0.setData(view0_filt)
    curve1.setData(view1_filt)
    
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
    
    # Update radial grid circles
    for i, circle_item in enumerate(radial_grid_circles):
        radius = r * (i + 1) / 4  # Divide into 4 sections
        circle_item.setRect(float(-radius), float(-radius), float(2*radius), float(2*radius))
    
    # Use filtered recent data for scatter plot (last 256 points)
    scatter_x = view0[-256:]
    scatter_y = view1[-256:]
    scatter.setData(scatter_x, scatter_y, brush=colors)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

app.exec()