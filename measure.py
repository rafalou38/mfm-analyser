import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import time
from scipy.signal import butter
from scipy.optimize import curve_fit
import time

from LockIn import LockIn

HISTORY = 256*30

lk = LockIn(HISTORY)

# Graphique Signaux(t)
class MainPlot:
    def __init__(self, win):
        self.plot = win.addPlot(title="Signaux",row=0, col=0)
        self.plot.enableAutoRange('x', True)
        self.plot.setYRange(0, 1)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve0 = self.plot.plot(pen=pg.mkPen('b', width=3), name="A0")
        self.curve1 = self.plot.plot(pen=pg.mkPen('r', width=3), name="Model")

        self.peaks = []
        self.peak_items = []

        self.model = None
        self.model_t = None

    def draw(self, t, x):
        self.curve0.setData(t, x)

        y_range_min = float(min(lk.y_min, lk.x_min) * 1.1)
        y_range_max = float(max(lk.x_max, lk.y_max) * 1.85)
        if np.isfinite([y_range_min, y_range_max]).all() and abs(y_range_max - y_range_min) < 1e6:
            self.plot.setYRange(y_range_min, y_range_max)

        # Mis a jour pics
        if self.model is not None:
            self.curve1.setData(self.model_t, self.model)
        else:
            self.curve1.setData([], [])

        for item in self.peak_items:
            self.plot.removeItem(item)
        self.peak_items.clear()
        # for e in self.peak_items:
        #     # Remove
        #     self.plot.removeItem(e)
        for peak_i in self.peaks:
            peak_t = t[peak_i]
            # print(f"Peak at {t:.2f} s")
            self.peak_items.append(pg.InfiniteLine(pos=peak_t, angle=90, pen='r'))
            self.plot.addItem(self.peak_items[-1])


class FFTView:
    def __init__(self, win):
        self.plot = win.addPlot(title="FFT", row=1, col=0)
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

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Application principale

app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="NanoSpeed")

cursor_view = CursorView(win)
main_plot = MainPlot(win)
# fft_view = FFTView(win)

main_plot.plot.getViewBox().sigResized.connect(lambda: cursor_view.update_geometry(main_plot.plot))
cursor_view.update_geometry(main_plot.plot)

def draw():
    x = np.roll(lk.buf0, -lk.ptr)
    y = np.roll(lk.buf1, -lk.ptr)
    t = np.roll(lk.buft, -lk.ptr)

    main_plot.draw(t, x)

    # if
    # print(lk.ready)
    # try:
    #     fft_view.setData(t, x)
    # except Exception as e:
    #     # print("Error in FFTView:", e)
    #     pass

    cursor_view.set_data(
        x[-256:],
        y[-256:],
    )
    cursor_view.draw(lk)

paused = False
ready = False
def update():
    global ready
    if not paused:
        lk.read_serial(phase=False)

    if ready or lk.ptr > 500:
        draw()
        ready = True

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

# Premiere approche
def log_deg():
    x = np.roll(lk.buf0, -lk.ptr)
    peaks = find_peaks(x, prominence=0.01, distance=50)
    print("Found peaks:", len(peaks[0]))
    main_plot.peaks = peaks[0]

# Modèle utilisé
def sin_exp_dec(t, X0, omega0, phi, Q, C):
    return X0 * np.exp(-t * omega0 / (2*Q)) * np.cos(omega0 * t * np.sqrt(1 - 1/(4*Q**2)) + phi) + C

popt = None
def model_dec():
    global popt
    x = np.roll(lk.buf0, -lk.ptr)
    t = np.roll(lk.buft, -lk.ptr)

    print("Fit en cours...")
    popt, pcov = curve_fit(
        sin_exp_dec,
        t - t[0],
        x,
        p0=[0.02, 2*np.pi*40, 0, 200, 0],
        bounds=(
            [0,    2*np.pi*20,  -np.pi,  10,   -0.02],
            [0.1, 2*np.pi*80,   np.pi, 500,    0.02],
        ),
        maxfev=3000,
        method='trf'
    )

    print("Résultats:")
    print(f"f0: {popt[1] / (2*np.pi):.1f} Hz")
    print(f"Q: {popt[3]:.1f}")
    print(f"k: {1.3e-3 * (popt[1])**2:.1f} alpha: {1.3e-3 * popt[1] / popt[3]:.1f}")

    main_plot.model = sin_exp_dec(t - t[0], *popt)
    main_plot.model_t = t

def save():
    global popt
    if popt is not None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        np.savez(f"{timestamp}-m.npz", popt=popt, x=np.roll(lk.buf0, -lk.ptr), t=np.roll(lk.buft, -lk.ptr))
        print("Model parameters saved.")
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        np.savez(f"{timestamp}-m.npz", x=np.roll(lk.buf0, -lk.ptr), t=np.roll(lk.buft, -lk.ptr))
        print("No model parameters to save.")

class KeyHandler(QtCore.QObject):
    def eventFilter(self, obj, event):
        global paused
        if event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() == QtCore.Qt.Key.Key_Space:
                if paused:
                    main_plot.model = None
                    main_plot.model_t = None
                paused = not paused
                return True
            if event.key() == QtCore.Qt.Key.Key_D:
                log_deg()
                return True
            if event.key() == QtCore.Qt.Key.Key_M:
                model_dec()
                paused = True
                return True
            if event.key() == QtCore.Qt.Key.Key_S:
                save()
                return True
            if event.key() == QtCore.Qt.Key.Key_R:
                lk.clear_serial()
                return True
            if event.key() == QtCore.Qt.Key.Key_F:
                lk.reset_bounds()
                lk.update_bounds()
                return True
        return super().eventFilter(obj, event)

key_handler = KeyHandler()
app.installEventFilter(key_handler)

app.exec()