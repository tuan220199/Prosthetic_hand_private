from PyQt5 import  QtWidgets, QtCore, QtGui
from gforce import  DataNotifFlags
import time
import torch
import random
import os
import numpy as np
from  pagewindow import PageWindow
from customcanvas import CustomFigCanvaswoRMS
import threading
from statistics import mode
from modeltraining import loadLogRegr,loadwoOperator, loadwithOperator,get_class
from helpers import set_cmd_cb, rms_formuula

channels = []

actions = list(range(1,10))*3
random.shuffle(actions)
actionLabelglobal = None

OFFSET = 121

sampRate = 500
channelMask = 0xFF
dataLen = 128
resolution = 8

ACTION = 0

reg = None
currentaction = "Flexion"
ACTIONS = {
    1: ["Flexion",          "img/Flexion.png",          (None, None),  0],
    2: ["Extension",        "img/Extension.png",        (None, None),  0],
    3: ["Ulnar Deviation",  "img/UlnarDeviation.png",   (None, None),  0],
    4: ["Radial Deviation", "img/RadialDeviation.png",  (None, None),  0],
    5: ["Supination",       "img/Supination.png",       (None, None),  0],
    6: ["Pronation",        "img/Pronation.png",        (None, None),  0],
    7: ["Open palm",        "img/Open.png",             (None, None),  0],
    8: ["Close palm",       "img/Close.png",            (None, None),  0],
    9: ["Rest",             "img/Rest.png",             (None, None),  0],
    }

packet_cnt = 0
start_time = 0
FORWARD = 0
ind_channel = 0

def ondata(data):
    global channels

        # Data for EMG CH0~CHn repeatly.
        # Resolution set in setEmgRawDataConfig:
        #   8: one byte for one channel
        #   12: two bytes in LSB for one channel.
        # eg. 8bpp mode, data[1] = channel[0], data[2] = channel[1], ... data[8] = channel[7]
        #                data[9] = channel[0] and so on
        # eg. 12bpp mode, {data[2], data[1]} = channel[0], {data[4], data[3]} = channel[1] and so on

        # # end for
        
    extracted_data = data[1:]
    channels += extracted_data

class EvaluationWindow(PageWindow):
    def __init__(self, GF, actionLabel):
        super().__init__()
        global actionLabelglobal
        self.initUI()
        self.GF = GF
        actionLabelglobal= actionLabel

    def initUI(self):
        self.setWindowTitle("Scan for device")
        self.setGeometry(100, 100, 1500, 900)
        self.UiComponents()

    def goToMain(self):
        self.goto("main")

    
    def start(self):
        self.GF.setEmgRawDataConfig(sampRate, channelMask, dataLen, resolution, cb=set_cmd_cb, timeout=1000)
        self.GF.setDataNotifSwitch(DataNotifFlags['DNF_EMG_RAW'], set_cmd_cb, 1000)
        self.GF.startDataNotification(ondata)

        self.setWindowTitle("Visualize EMG Data")
        self.layout1.addWidget(self.skipSignalButton)
        self.layout1.addWidget(self.trainModelButton)

        actionLabelglobal.setText('Select a model')
        actionLabelglobal.setFont(QtGui.QFont('Arial', 20))
        actionLabelglobal.setFixedSize(300,30)
        actionLabelglobal.setAlignment(QtCore.Qt.AlignCenter)
        self.layout5.addWidget(actionLabelglobal)
        self.layout0.addLayout(self.layout5)
        self.layout.addLayout(self.layout3)
        QtWidgets.qApp.processEvents()

        while True:
            if len(channels)>128: 
                break
        self.myFig = CustomFigCanvaswoRMS()
        self.layout.addWidget(self.myFig)
        #Add the callbackfunc to ..
        self.myDataLoop = threading.Thread(name = 'myDataLoop', target = dataSendLoop, daemon = True, args = (self.addData_callbackFunc,))
        self.myDataLoop.start()

    def resume(self):
        self.myDataLoop.join()

    def make_handleButton(self, button, *args):
        def handleButton():
            global reg, ind_channel, ACTION, REP, PEAK, PEAK_MULTIPLIER, OFFSET, STARTED, BASELINE, BASELINE_MULTIPLIER
            global OFFSET_RMS, file1,actionLabelglobal, FORWARD
            
            if button == "updateMotion":
                try:
                    self.loadNewAction(int(self.subj_motion.text()))
                except Exception as e:
                    print("Error during update motion: ", e)

            elif button=='backToCollect':
                reg = None
                self.trainModelButton.setEnabled(False)
                actionLabelglobal.setText('Select a model')
                QtWidgets.qApp.processEvents()

            elif button == "skipSignal":
                FORWARD += 1000
            
            elif button == "loadLogRegr":
                reg = loadLogRegr()
                self.trainModelButton.setEnabled(True)
            
            elif button == "loadwoOperator":
                reg = loadwoOperator()
                self.trainModelButton.setEnabled(True)
            
            elif button == "loadwithOperator":
                reg = loadwithOperator()
                self.trainModelButton.setEnabled(True)
            elif button == "menu":
                self.goto('main1')

        return handleButton
    
    def addData_callbackFunc(self, value):
        # print("Add data: " + str(value))
        self.myFig.addData(value)

    def UiComponents(self):
        global actionLabelglobal
        self.layout = QtWidgets.QVBoxLayout()
        self.layout0 = QtWidgets.QHBoxLayout()

        self.layout1 = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(10,10,10,10)
        self.layout.setSpacing(10)
        self.scanButton = QtWidgets.QPushButton("Back to Menu")
        self.scanButton.setFixedSize(100,30)
        self.scanButton.clicked.connect(self.make_handleButton("menu"))

        
        self.layout1.addWidget(self.scanButton)
        self.layout1.setAlignment(QtCore.Qt.AlignTop)

        self.layout0.addLayout(self.layout1)
        self.layout0.setAlignment(QtCore.Qt.AlignTop)
        self.layout.addLayout(self.layout0)

        
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        widget = QtWidgets.QWidget()
        widget.setLayout(self.layout)

        self.setCentralWidget(widget)

        self.layout2 = QtWidgets.QHBoxLayout()
        self.layout3 = QtWidgets.QHBoxLayout()

        b1 = QtWidgets.QPushButton("Logistic Regression")
        b1.clicked.connect(self.make_handleButton("loadLogRegr"))
        b1.setFixedSize(150,30)

        b2 = QtWidgets.QPushButton("Model without operator")
        b2.clicked.connect(self.make_handleButton("loadwoOperator"))
        b2.setFixedSize(150,30)

        b3 = QtWidgets.QPushButton("Model with operator (Recommended)")
        b3.clicked.connect(self.make_handleButton("loadwithOperator"))
        b3.setFixedSize(300,30)

        self.layout3.addWidget(b1)
        self.layout3.addWidget(b2)
        self.layout3.addWidget(b3)

        self.trainModelButton = QtWidgets.QPushButton("Unload model")
        self.trainModelButton.clicked.connect(self.make_handleButton("backToCollect"))
        self.trainModelButton.setFixedSize(150,30)
        self.trainModelButton.setEnabled(False)

        
        self.skipSignalButton = QtWidgets.QPushButton("Refresh")
        self.skipSignalButton.clicked.connect(self.make_handleButton("skipSignal"))
        self.skipSignalButton.setFixedSize(150,30)
        
        self.subj_name = QtWidgets.QLineEdit("1")
    

        self.layout5 = QtWidgets.QVBoxLayout()
        self.layout5.setAlignment(QtCore.Qt.AlignCenter)
        self.layout5.setContentsMargins(0, 0, 0, 0)

        

class Communicate(QtCore.QObject):
    data_signal = QtCore.pyqtSignal(list)


def dataSendLoop(addData_callbackFunc):
    # Setup the signal-slot mechanism.
    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)
    #time.sleep(3)
    global actionLabelglobal, ACTIONS,FORWARD, reg
    while(True):
        #channels[i:i+50*8]
        predictedclasses = []
        for j in range (15):
            
            try:
                datawindow = channels[FORWARD:FORWARD+50*8]
                if datawindow:
                    datastack = np.stack([np.array(datawindow[k::8]) for k in range (8)]).astype('float32') - OFFSET
                    #mean_in_window = datastack.mean(1) # should have size (8,)
                    rms_ = rms_formuula(datastack/255)
                    if reg:
                        pred_class = get_class(reg, torch.tensor(rms_.reshape(-1,8))) 
                        predictedclasses.append(pred_class)
                        mySrc.data_signal.emit( list(datastack.mean(1)))
                    else:
                        mySrc.data_signal.emit(list(datastack.mean(1)))
                    FORWARD += 25*8 
                time.sleep(50/1000)
            

            except Exception as e:
                print("Error during plotting:", type(e),e) 
        if predictedclasses:
            smoothenedclass = mode(predictedclasses)
            actionLabelglobal.setText(f'{ACTIONS[smoothenedclass+1][0]}')
