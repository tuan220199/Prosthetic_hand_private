from pagewindow import PageWindow
from PyQt5 import QtCore, QtWidgets

class MainWindow(PageWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(300, 300, 800, 400)
        self.initUI()
        self.setWindowTitle("Main Window")
        
        

    def initUI(self):
        self.UiComponents()

    def UiComponents(self):
        self.searchButton = QtWidgets.QPushButton("Start", self)
        self.searchButton.setGeometry(QtCore.QRect(5, 5, 200, 40))
        self.searchButton.clicked.connect(
            self.make_handleButton("searchButton")
        )
    def make_handleButton(self, button, *args):
        def handleButton():
            self.goto("search")
        return handleButton
