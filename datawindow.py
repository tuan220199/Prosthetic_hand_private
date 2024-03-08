from pagewindow import PageWindow
from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, mkPen
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from gforce import NotifDataType, DataNotifFlags
import threading
import time
import random


class DataWindow(PageWindow):
    def __init__(self, GF):
        super().__init__()
        self.GF = GF
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle("EMG data")
        self.UiComponents()
    
    def UiComponents(self):
        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle("my first window")
        # Create FRAME_A
        self.FRAME_A = QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QColor(210,210,235,255).name())
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)

        # Place the matplotlib figure
        self.graphWidget = PlotWidget()
        self.LAYOUT_A.addWidget(self.graphWidget, *(0,1))

        self.x = list(range(100))
        self.y = list(range(0,200,2))
        pen = mkPen(color = (255,0,0))
        self.data_line = self.graphWidget.plot(self.x, self.y, pen=pen)
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()
        #self.myFig = CustomFigCanvas()
        #self.LAYOUT_A.addWidget(self.myFig, *(0,1))
        # Add the callbackfunc to ..
        #myDataLoop = threading.Thread(name = 'myDataLoop', target = dataSendLoop, daemon = True, args = (self.addData_callbackFunc,))

        #myDataLoop.start()
        
        
        
    def addData_callbackFunc(self, value):
        # print("Add data: " + str(value))
        self.myFig.addData(value)
    def update_plot_data(self):
        self.x = self.x[1:]
        self.x.append(self.x[-1] +1)
        self.y = self.y[1:]
        self.y.append(random.randint(0,100))
        self.data_line.setData(self.x, self.y)

class Communicate(QObject):
    data_signal = pyqtSignal(list)

def dataSendLoop(addData_callbackFunc):
    # Setup the signal-slot mechanism.
    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)

    i = 0
    while(True):
        mySrc.data_signal.emit([channels[i:i+8],ACTION*10+100]) 
        time.sleep(0.1)
        i += 1