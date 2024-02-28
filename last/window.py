from PyQt5 import QtCore, QtWidgets

from  pagewindow import PageWindow
from mainwindow import MainWindow
from datacollection import DataCollectionhWindow
from evaluationwindow import EvaluationWindow
from gforce import GForceProfile

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.GF = GForceProfile()
         
        self.stacked_widget = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.m_pages = {}

        self.register(MainWindow(self.GF), "main")
        self.register(DataCollectionhWindow(self.GF), "dataCollect")
        self.register(EvaluationWindow(self.GF,  QtWidgets.QLabel()), "evaluate")

        self.goto("main0")

    def register(self, widget, name):
        self.m_pages[name] = widget
        self.stacked_widget.addWidget(widget)
        if isinstance(widget, PageWindow):
            widget.gotoSignal.connect(self.goto)
    def addData_callbackFunc(self, value):
        self.myFig.addData(value)

    @QtCore.pyqtSlot(str)
    def goto(self, name):
        comeback = int(name[-1])
        name = name[:-1]
        if name in self.m_pages:
            widget = self.m_pages[name]
            
            self.stacked_widget.setCurrentWidget(widget)
            if comeback:
                widget.resume()
            else:
                widget.start()
            self.setWindowTitle(widget.windowTitle())