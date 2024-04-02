from Packages import *
from Generator import PointProcess

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None) -> None:
        """
        Initialize the canvas with two subplots

        Parameters
        ----------
        - parent : QMainWindow, optional
            Parent widget, by default None
        
        Returns
        -------
        - None
        """
        fig, (self.axe1, self.axe2) = plt.subplots(nrows=2, sharex=True)
        fig.tight_layout()
        self.format_plot()
        super(MplCanvas, self).__init__(fig)
        return None
    
    def format_plot(self) -> None:
        """
        Format the plot with grid and labels

        Parameters:
        -----------
        - None

        Returns
        -------
        - None
        """
        self.axe1.grid(True)
        self.axe2.grid(True)

        self.axe2.set_xlabel('Time (s)')

        self.axe1.set_ylabel('Ut')
        self.axe2.set_ylabel('Xt')
        return None

class MainWindow(QMainWindow):

    def __init__(self, mu:float, T:float, *args, **kwargs) -> None:
        """
        Initialize the main window with the canvas and the timer

        Parameters
        ----------
        - mu : float
            Initial intensity of the Poisson process
        - T : float
            Time of simulation
        
        Returns
        -------
        - None
        """
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowState(Qt.WindowMaximized)

        self.canvas = MplCanvas(self)
        self.canvas.setGeometry(0, 0, 1920, 1080)
        self.setCentralWidget(self.canvas)
        self.canvas.setStyleSheet("background-color: black;")

        self.deltatime = 10
        self.show()

        self.mu = mu
        self.T = T
        self.generate = PointProcess(self.mu, self.T)
        self.s, self.U, self.X = 0, 0, 0
        self.s, self.U, self.X = self.generate.simulate_realtime(self.s, self.U, self.X)
        self.times, self.Ulist, self.Xlist = np.array([0]), np.array([0]), np.array([0])
       
        self.timer = QTimer()
        self.timer.setInterval(self.deltatime)
        self.timer.timeout.connect(self.update_plot)
        self.init_time = timeit.default_timer()
        self.timer.start()
        return None

    def update_plot(self) -> None:
        """
        Update the plot with the new values of the Poisson process

        Parameters:
        -----------
        - None

        Returns
        -------
        - None
        """
        self.canvas.axe1.clear()
        self.canvas.axe2.clear()
        self.canvas.format_plot()

        current_time = timeit.default_timer() - self.init_time
        if current_time > self.T:
            self.timer.stop()
            return None
        
        if current_time < self.s:
            self.times = np.append(self.times, current_time)
            self.Ulist = np.append(self.Ulist, self.Ulist[-1])
            self.Xlist = np.append(self.Xlist, self.Xlist[-1])
        else:
            self.times = np.append(self.times, self.s)
            self.Ulist = np.append(self.Ulist, self.U)
            self.Xlist = np.append(self.Xlist, self.X)
            self.s, self.U, self.X = self.generate.simulate_realtime(self.s, self.U, self.X)
        
        self.canvas.axe1.step(self.times, self.Ulist, 'black')
        self.canvas.axe2.step(self.times, self.Xlist, 'black')

        self.canvas.draw()
        return None

if __name__ == "__main__":
    app = QApplication(sys.argv)

    f = open('config.json', 'r')
    data = json.load(f)
    mu = data["config_app"]["mu"]
    T = data["config_app"]["T"]
    f.close()

    w = MainWindow(mu, T)
    app.exec()
    pass
