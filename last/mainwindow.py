from pagewindow import PageWindow
from PyQt5 import QtCore, QtWidgets

class MainWindow(PageWindow):
    def __init__(self, GF):
        super().__init__()
        self.initUI()
        self.GF = GF
        self.devices = []
        self.visitedDataCollect = 0
        self.vistedEvaluation = 0

    def initUI(self):
        self.setWindowTitle("Scan for device")
        self.setGeometry(100, 100, 1500, 900)
        self.UiComponents()
    def start(self):
        return

    def resume(self):
        return


    def UiComponents(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.layout0 = QtWidgets.QHBoxLayout()

        self.layout1 = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(10,10,10,10)
        self.layout.setSpacing(10)
        self.scanButton = QtWidgets.QPushButton("Scan")
        self.scanButton.setFixedSize(100,30)
        self.scanButton.clicked.connect(self.make_handleButton("scan"))
        self.l1 = QtWidgets.QLabel()
        self.l1.setText("Click Scan to start scanning")
        self.l1.setFixedSize(300,30)
        
        self.layout1.addWidget(self.scanButton)
        self.layout1.addWidget(self.l1)
        self.layout1.setAlignment(QtCore.Qt.AlignTop)

        self.layout0.addLayout(self.layout1)
        self.layout0.setAlignment(QtCore.Qt.AlignTop)
        self.layout.addLayout(self.layout0)

        
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        widget = QtWidgets.QWidget()
        widget.setLayout(self.layout)

        self.setCentralWidget(widget)

        self.layout2 = QtWidgets.QVBoxLayout()
        self.layout3 = QtWidgets.QHBoxLayout()
        self.dataCollectButton = QtWidgets.QPushButton("Data Collection")
        self.dataCollectButton.setFixedSize(200,200)
        self.dataCollectButton.clicked.connect(self.make_handleButton("dataCollect"))

        self.evaluationButton = QtWidgets.QPushButton("Evaluation")
        self.evaluationButton.setFixedSize(200,200)
        self.evaluationButton.clicked.connect(self.make_handleButton("evaluate"))



    def scan(self):
        scan_results = self.GF.scan(2)

        if scan_results:
            self.l1.setText(f"Found {len(scan_results)}")
            for result in scan_results:
                devButton = QtWidgets.QPushButton(f"{result[2]}")
                devButton.clicked.connect(self.make_handleButton("connectToDevice", result[2]))
                self.layout2.addWidget(devButton)
            self.layout0.addLayout(self.layout2)   
            self.layout2.setAlignment(QtCore.Qt.AlignTop)

        else:
            self.l1.setText("No bracelet was found")
    
    def connect(self,*args):
        try:
            self.GF.connect(addr=args[0])
        except:
            self.l1.setText(f"Can not conect to address {args[0]}. Please scan again.")
        while self.layout2.count():
            child = self.layout2.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        QtWidgets.qApp.processEvents()

        self.l1.setText(f"Connected to {args[0]}")
        self.layout3.addWidget(self.dataCollectButton)
        self.layout3.addWidget(self.evaluationButton)
        self.layout.addLayout(self.layout3)

    def make_handleButton(self, button, *args):
        def handleButton():
            if button == "scan":
                self.l1.setText("Scanning...")
                QtWidgets.qApp.processEvents()
                self.scan()
                self.scanButton.setText("Scan Again")
            
            elif button == "connectToDevice":
                self.connect(*args)

            elif button == "dataCollect":
                print('Going to Data Collection')
                self.goto(f"dataCollect{self.visitedDataCollect}")
                self.visitedDataCollect = 1
            elif button == "evaluate":
                print('Going to Evaluation')

                self.goto(f"evaluate{self.vistedEvaluation}") 
                self.vistedEvaluation = 1
        return handleButton
