from PyQt5 import QtWidgets, QtCore
from  pagewindow import PageWindow
import time
from gforce import  DataNotifFlags

def set_cmd_cb(resp):
    print('Command result: {}'.format(resp))
file1 = open("myfile.txt","w")
channels = []
ACTION = 0

def ondata(data):
    global ACTION
    global channels

    # Data for EMG CH0~CHn repeatly.
    # Resolution set in setEmgRawDataConfig:
    #   8: one byte for one channel
    #   12: two bytes in LSB for one channel.
    # eg. 8bpp mode, data[1] = channel[0], data[2] = channel[1], ... data[8] = channel[7]
    #                data[9] = channel[0] and so on
    # eg. 12bpp mode, {data[2], data[1]} = channel[0], {data[4], data[3]} = channel[1] and so on

    extracted_data = data[1:]
    channels += extracted_data
    file1.write(', '.join(map(str, extracted_data)) +', ' + str(ACTION) +"\n")

    global packet_cnt
    global start_time

    if start_time == 0:
        start_time = time.time()
    
    packet_cnt += 1
    
    if time.time() - start_time > 5:
        ACTION += 1
        print(', '.join(map(str, extracted_data)))
        print('perform action', ACTION, '\n')
        start_time = time.time()

class MenuWindow(PageWindow):
    def __init__(self, GF):
        super().__init__()
        self.setGeometry(300, 300, 800, 400)
        self.initUI()
        self.setWindowTitle("Function")
        self.GF = GF
        self.sampRate = 500
        self.channelMask = 0xFF
        self.dataLen = 128
        self.resolution = 8 
        

    def initUI(self):
        self.UiComponents()

    def make_handleButton(self, button, *args):
        def handleButton():
            if button == "goToRecord":
                self.GF.setEmgRawDataConfig(self.sampRate, self.channelMask, self.dataLen, self.resolution, cb=set_cmd_cb, timeout=1000)
                self.GF.setDataNotifSwitch(DataNotifFlags['DNF_EMG_RAW'], set_cmd_cb, 1000)
                self.GF.startDataNotification(ondata)
                self.goto("dataVisualization")        
            
            elif button == "searchButton":
                self.goto("search")
                     
        return handleButton
    
    def UiComponents(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.backButton = QtWidgets.QPushButton("Back")
        self.backButton.clicked.connect(self.make_handleButton("searchButton"))
        
        self.dataRecord = QtWidgets.QPushButton("Get Raw EMG data")
        self.dataRecord.clicked.connect(self.make_handleButton("goToRecord"))
        self.layout.addWidget(self.backButton)
        self.layout.addWidget(self.dataRecord)
        widget = QtWidgets.QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        self.setLayout(self.layout)