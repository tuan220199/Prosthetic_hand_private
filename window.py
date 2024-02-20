from PyQt5 import QtCore, QtWidgets

from  pagewindow import PageWindow
from mainwindow import MainWindow
from searchwindow import SearchWindow
from menuwindow import MenuWindow
from datawindow import DataWindow
from gforce import GForceProfile

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.GF = GForceProfile()
         
        self.stacked_widget = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.m_pages = {}

        self.register(MainWindow(), "main")
        self.register(SearchWindow(self.GF), "search")
        self.register(MenuWindow(self.GF), "menu")
        self.register(DataWindow(self.GF), "dataVisualization")

        self.goto("main")

    def register(self, widget, name):
        self.m_pages[name] = widget
        self.stacked_widget.addWidget(widget)
        if isinstance(widget, PageWindow):
            widget.gotoSignal.connect(self.goto)

    @QtCore.pyqtSlot(str)
    def goto(self, name):
        if name in self.m_pages:
            widget = self.m_pages[name]
            self.stacked_widget.setCurrentWidget(widget)
            self.setWindowTitle(widget.windowTitle())