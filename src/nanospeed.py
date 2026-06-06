import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from LockIn import LockIn

HISTORY = 256*4

FFT_ONLY = False

lk = LockIn(HISTORY)

# Graphique Signaux(t)
class MainPlot:
    def __init__(self, win):
        self.plot = win.addPlot(title="Signaux",row=0, col=0)
        self.plot.enableAutoRange('x', False)
        self.plot.setYRange(0, 1)
        self.plot.setXRange(0, 0.25)

        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve0 = self.plot.plot(pen=pg.mkPen('b', width=3), name="A0")
        self.curvez = self.plot.plot(pen=pg.mkPen('m', width=3), name="Az")

        self.peak_items = []

    def draw(self, t, x, y, ttl, lk, s):
        t0 = lk.get_trigger()
        # Affichage courbes
        self.curve0.setData(t - t0, x)
        self.curvez.setData(t - t0, ttl * max(lk.x_max, lk.y_max))

        # Mise a jour range
        y_range_min = float(min(lk.y_min, lk.x_min) * 2)
        y_range_max = float(max(lk.x_max, lk.y_max) * 5)
        if np.isfinite([y_range_min, y_range_max]).all() and abs(y_range_max - y_range_min) < 1e6:
            self.plot.setYRange(y_range_min, y_range_max)

        # Mis a jour pics
        for item in self.peak_items:
            self.plot.removeItem(item)
        for peak_t in lk.peaks:
            self.peak_items.append(pg.InfiniteLine(pos=peak_t-t0, angle=90, pen='r'))
            self.plot.addItem(self.peak_items[-1])

# Graphique Phase(t)
class PhasePlot:
    def __init__(self, win):
        self.plot = win.addPlot(title="Phase",row=1, col=0)
        self.plot.enableAutoRange('x', True)
        self.plot.enableAutoRange('y', True)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve0 = self.plot.plot(name="A0", pen=pg.mkPen('b', width=3))

    def draw(self, phi, phi_t):
        self.curve0.setData(phi_t, np.degrees(phi))

# Graphique Amplitude(t)
class AmpPlot:
    def __init__(self, win):
        self.plot = win.addPlot(title="Amplitude",row=2, col=0)
        self.plot.enableAutoRange('x', True)
        self.plot.enableAutoRange('y', True)

        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve0 = self.plot.plot(name="A0", pen=pg.mkPen('r', width=3))

    def draw(self, amp, amp_t):
        self.curve0.setData(amp_t, amp)

class FFTView:
    def __init__(self, win):
        self.plot = win.addPlot(title="FFT", row=1, col=0)
        self.plot.enableAutoRange(True)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen='w')
        self.i = 0
        self.labels = []

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
        overlay.setBackgroundColor((255, 255, 255, 200))
        win.scene().addItem(overlay)

        self._overlay = overlay

        # Grille radiale en fond
        self.radial_grid_circles = []
        for i in range(1, 5):
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


        # Graphique xy
        self.scatter = pg.ScatterPlotItem(size=4, brush='r', pen=None)

        overlay.addItem(self.scatter)
        overlay.addItem(vline)
        overlay.addItem(hline)
        overlay.addItem(self.circle)

        # Dégradé de couleurs d'ancienneté
        alpha = np.linspace(255, 0, 256).astype(np.uint8)
        self.colors = np.zeros((256, 4), dtype=np.uint8)
        self.colors[:,0] = 255
        self.colors[:,3] = alpha

    def draw(self, lk):
        # Calcul échelle
        r = float(max(abs(lk.x_min), abs(lk.x_max), abs(lk.y_min), abs(lk.y_max)))
        if not np.isfinite(r) or r > 10 or r < 1e-10:
            return
        
        self.scale_text.setText(f"±{r:.3f}")
        self.scale_text.setPos(r, r)

        self._overlay.setXRange(float(-r), float(r))
        self._overlay.setYRange(float(-r), float(r))

        self.circle.setRect(float(-r), float(-r), float(2*r), float(2*r))
        
        # Mise a jour grille
        for i, circle_item in enumerate(self.radial_grid_circles):
            radius = r * (i + 1) / 4
            circle_item.setRect(float(-radius), float(-radius), float(2*radius), float(2*radius))
    
    def update_geometry(self, plot):
        rect = plot.getViewBox().sceneBoundingRect()

        w = rect.height() * 0.36
        h = rect.height() * 0.35

        self._overlay.setGeometry(
            rect.x() + rect.width() - w,
            rect.y(),
            w,
            h
        )

    def set_data(self, x, y):
        self.scatter.setData(x, y, brush=self.colors)

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Application principale

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
    x,y,ttl,t, phi, phi_t, amp, s,phi_simple, phi_simple_t, amp_simple = lk.get_raw_data()

    if FFT_ONLY:
        fft_view.setData(t, x)
    else:
        mask = phi_simple_t > 0
        amp_plot.draw(amp_simple[mask], phi_simple_t[mask])
        phase_plot.draw(phi_simple[mask], phi_simple_t[mask])

    main_plot.draw(t, x, y, ttl, lk, s)

    cursor_view.set_data(
        x[-256:],
        y[-256:],
    )
    cursor_view.draw(lk)

saving = False
def update():
    if saving: return
    lk.read_serial()

    if lk.ready:
        draw()

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

class KeyHandler(QtCore.QObject):
    def eventFilter(self, obj, event):
        global saving
        if event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() == QtCore.Qt.Key.Key_R:
                lk.reset_phase()
                return True
            if event.key() == QtCore.Qt.Key.Key_Space:
                saving = not saving
                return True
        return super().eventFilter(obj, event)
key_handler = KeyHandler()
app.installEventFilter(key_handler)

app.exec()