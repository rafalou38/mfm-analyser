import serial
import numpy as np
import matplotlib.pyplot as plt


# Affichage des données brutes de l'arduino nano


ser = serial.Serial('/dev/ttyUSB1', 1000000, timeout=0)

plt.ion()

fig, ax = plt.subplots()

line0, = ax.plot([], [], label="A0")
line1, = ax.plot([], [], label="A1")

ax.set_ylim(0, 4095)
ax.legend()

while True:
    data = ser.read(1024)
    samples = np.frombuffer(data, dtype=np.uint16)

    ch0 = samples[0::2]
    ch1 = samples[1::2]

    line0.set_data(range(len(ch0)), ch0)
    line1.set_data(range(len(ch1)), ch1)

    ax.set_xlim(0, len(ch0))

    fig.canvas.draw()
    fig.canvas.flush_events()