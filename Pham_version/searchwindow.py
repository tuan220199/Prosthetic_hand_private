from pagewindow import PageWindow
from PyQt5 import QtWidgets
from gforce import GForceProfile


class SearchWindow(PageWindow):
    def __init__(self, GF):
        super().__init__()
        self.initUI()
        self.GF = GF

    def initUI(self):
        self.setWindowTitle("Scan for device")
        self.UiComponents()

    def goToMain(self):
        self.goto("main")

    def make_handleButton(self, button, *args):
        def handleButton():
            if button == "scan":
                self.l1.setText("Scanning...")
                QtWidgets.qApp.processEvents()
                
                scan_results = self.GF.scan(5)
                if scan_results:
                    self.l1.setText(f"Found {len(scan_results)}")
                    for result in scan_results:
                        devButton = QtWidgets.QPushButton(f"{result}")
                        devButton.clicked.connect(self.make_handleButton("connectToDevice", result[2]))
                        self.layout.addWidget(devButton)
                else:
                    self.l1.setText("No bracelet was found")
                self.scanButton.setText("Scan Again")       
            
            elif button == "connectToDevice":
                try:
                    self.GF.connect(addr=args[0])
                    
                    
                except:
                    self.l1.setText(f"Can not conect to address {args[0]}. Please scan again.")            
        return handleButton
    
    def UiComponents(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.backButton = QtWidgets.QPushButton("Back")
        self.backButton.clicked.connect(self.goToMain)

        self.scanButton = QtWidgets.QPushButton("Scan")
        self.scanButton.clicked.connect(self.make_handleButton("scan"))

        self.l1 = QtWidgets.QLabel()
        self.l1.setText("Click Scan to start scanning")
        
        self.layout.addWidget(self.backButton)
        self.layout.addWidget(self.scanButton)
        self.layout.addWidget(self.l1)

        widget = QtWidgets.QWidget()
        widget.setLayout(self.layout)

        self.setCentralWidget(widget)
