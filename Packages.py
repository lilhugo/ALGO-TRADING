import sys
import json
import matplotlib
from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.interpolate import interp1d
import timeit

matplotlib.use('Qt5Agg')
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': 'Times New Roman'})