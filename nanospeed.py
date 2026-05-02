import serial
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import time
import subprocess
from scipy.signal import butter, filtfilt, sosfilt, sosfilt_zi

from LockIn import LockIn

HISTORY = 10000          # visible history

FFT_ONLY = False

################################
################################
### SIGNALS
################################
################################

# ############
# Data
# ############

lk = LockIn(HISTORY)

# Trigger settings
filter_enabled = False
filt_b, filt_a = butter(3, 0.15, btype='low')


trigger_enabled = False
fft_enabled = False
trigger_channel = 0  # 0 for buf0, 1 for buf1
trigger_level = 0.1
trigger_offset = 0  # Offset from trigger point



class MainPlot:
    def __init__(self, win):
        self.plot = win.addPlot(title="Signals",row=0, col=0)#, colspan=2 )
        self.plot.enableAutoRange('x', True)
        self.plot.setYRange(0, 1)
        self.plot.showGrid(x=True, y=True, alpha=0.3)  # Add grid to main plot
        self.curve0 = self.plot.plot(pen='y', name="A0")
        self.curve1 = self.plot.plot(pen='c', name="A1")
        self.curvez = self.plot.plot(pen='m', name="Az")
        self.curvec = self.plot.plot(pen='m', name="Az")

        self.peak_items = []

        

    def draw(self, t, x, y, ttl, lk):
        self.curve0.setData(t - t[0], x)
        self.curve1.setData(t - t[0], y)
        self.curvez.setData(t - t[0], ttl * max(lk.x_max, lk.y_max))

        y_range_min = float(min(lk.y_min, lk.x_min) * 2)
        y_range_max = float(max(lk.x_max, lk.y_max) * 5)
        if np.isfinite([y_range_min, y_range_max]).all() and abs(y_range_max - y_range_min) < 1e6:
            self.plot.setYRange(y_range_min, y_range_max)
class PhasePlot:
    def __init__(self, win):
        self.plot = win.addPlot(title="Phase",row=1, col=0)
        self.plot.enableAutoRange('x', True)
        self.plot.enableAutoRange('y', True)
        # self.plot.setYRange(-np.pi, np.pi)
        self.plot.showGrid(x=True, y=True, alpha=0.3)  # Add grid to main plot
        self.curve0 = self.plot.plot(pen='y', name="A0")
        

    def draw(self, phi, phi_t):
        self.curve0.setData(phi_t, phi)
class AmpPlot:
    def __init__(self, win):
        self.plot = win.addPlot(title="Amplitude",row=2, col=0)
        self.plot.enableAutoRange('x', True)
        self.plot.enableAutoRange('y', True)
        # self.plot.setYRange(-np.pi, np.pi)
        self.plot.showGrid(x=True, y=True, alpha=0.3)  # Add grid to main plot
        self.curve0 = self.plot.plot(pen='y', name="A0")
        

    def draw(self, amp, amp_t):
        self.curve0.setData(amp_t, amp)

class FFTView:
    def __init__(self, win):
        self.plot = win.addPlot(title="FFT", row=1, col=1)
        self.plot.enableAutoRange(True)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen='w')
        self.i = 0
        self.labels = []

    # def update():

    def setData(self, t, x):
        if np.isnan(x).any(): 
            print("nan bad")
            exit(1)
            return

        sp = rfft(x)
        dt = abs(np.median(np.diff(t)))
        self.i+=1
        if self.i % 100 ==0:
            print("Sampling: ", round(1/dt), "Hz")
        freqs = rfftfreq(len(x), d=dt)
        magnitude = np.abs(sp)
        magnitude /= np.max(magnitude)

        f_view = freqs[freqs<500]
        m_view = magnitude[freqs<500]

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
    def draw(self, lk):
        # Update overlay pointer
        r = float(max(abs(lk.x_min), abs(lk.x_max), abs(lk.y_min), abs(lk.y_max)))
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
if FFT_ONLY:
    fft_view = FFTView(win)
else:
    phase_plot = PhasePlot(win)
    amp_plot = AmpPlot(win)

main_plot.plot.getViewBox().sigResized.connect(lambda: cursor_view.update_geometry(main_plot.plot))
cursor_view.update_geometry(main_plot.plot)

def draw():
    x,y,ttl,t, phi, phi_t, amp = lk.get_raw_data()

    if FFT_ONLY:
        fft_view.setData(t, x)
    else:
        phase_plot.draw(phi, phi_t)
        # amp_plot.draw(amp,phi_t)

    main_plot.draw(t, x, y, ttl, lk)


    cursor_view.set_data(
        x[-256:],
        y[-256:],
    )
    cursor_view.draw(lk)

def update():
    lk.read_serial(handle_phase=not FFT_ONLY)

    if lk.ready:
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