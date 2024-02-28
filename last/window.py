from PyQt5 import QtCore, QtWidgets

from  pagewindow import PageWindow
from mainwindow import MainWindow
from datacollection import DataCollectionhWindow
from evaluationwindow import EvaluationWindow
from gforce import GForceProfile

class Window(QtWidgets.QMainWindow):
    """
    Class representing the main window of the application.

    This window manages a stacked widget for switching between different pages.
    Each page is registered and stored in a dictionary for easy access.

    Attributes:
        stacked_widget (QtWidgets.QStackedWidget): Stacked widget for managing pages.
        m_pages (dict): Dictionary to store registered pages.
        GF (GForceProfile): Instance of GForceProfile for data management.

    Signals:
        gotoSignal (str): Signal emitted when switching between pages.

    Methods:
        __init__(self, parent=None):
            Initializes the main window.

        register(self, widget, name):
            Registers a page widget with a given name.

        addData_callbackFunc(self, value):
            Callback function to add data to the figure.

        goto(self, name):
            Navigates to the specified page.
    """
    def __init__(self, parent=None):
        """
        Initializes the main window.

        Args:
            parent: Parent widget (default: None).
        """
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
        """
        Registers a page widget with a given name.

        Args:
            widget: Page widget to be registered.
            name (str): Name of the page.
        """
        self.m_pages[name] = widget
        self.stacked_widget.addWidget(widget)
        if isinstance(widget, PageWindow):
            widget.gotoSignal.connect(self.goto)
    def addData_callbackFunc(self, value):
        """
        Callback function to add data to the figure.

        Args:
            value: Data value to be added.
        """
        self.myFig.addData(value)

    @QtCore.pyqtSlot(str)
    def goto(self, name):
        """
        Navigates to the specified page.

        Args:
            name (str): Name of the page to navigate to.
        """
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