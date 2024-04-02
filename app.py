import sys
import random
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection

import numpy as np
matplotlib.use('Qt5Agg')

from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtCore import QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.S0 = 100
        self.deltat = 1 / 1e5
        self.sigma = 0.2

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.setCentralWidget(self.canvas)
        # set the background color of the canvas
        self.canvas.setStyleSheet("background-color: black;")

        # Create a range of data from 0 to 50
        self.xdata = np.array(range(-49, 1))
        self.ydata = np.array(np.ones(50) * self.S0)
        self._plot_ref = None
        self.deltatime = 100
        self.update_plot()

        self.show()

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QTimer()
        self.timer.setInterval(self.deltatime)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        # Drop off the first y element, append a new one.
        # self.yold = self.ydata
        self.ydata = np.roll(self.ydata, -1)
        self.ydata[-1] = self.ydata[-2] * (1 + 0.05 * self.deltat + self.sigma * np.sqrt(self.deltat) * np.random.normal(0,1))
        self.xdata = np.roll(self.xdata, -1)
        self.xdata[-1] = self.xdata[-2] + 1 * np.random.randint(1,3)
        self.canvas.axes.cla()  # Clear the canvas

        # mask_greater = np.where(self.ydata > self.yold)
        # mask_lower = np.where(self.ydata < self.yold)

        # diff = self.ydata - self.yold

        #segments = [((i, self.yold[i]), (i+1, self.ydata[i])) for i in range(len(self.ydata)-1)]
        #cmap = ListedColormap(['r', 'g',])
        #norm = BoundaryNorm([-10, 0, 10], cmap.N)
        #lc = LineCollection(segments, cmap=cmap, norm=norm)
        #lc.set_array(diff)
        #lc.set_linewidth(2)
        #line = self.canvas.axes.add_collection(lc)
        self.canvas.axes.autoscale()
        
        # self.canvas.axes.set_facecolor('black')
        self.canvas.axes.step(self.xdata,self.ydata)
        self.canvas.axes.grid(True, color='gray')
        self.canvas.axes.set_title('Live Updating stock')
        self.canvas.axes.set_xlabel('Time')
        self.canvas.axes.set_ylabel('Random Number')
        self.canvas.axes.tick_params(axis='x', colors='black')
        self.canvas.axes.tick_params(axis='y', colors='black')


        # Trigger the canvas to update and redraw.
        self.canvas.draw()

app = QApplication(sys.argv)
w = MainWindow()
app.exec()

