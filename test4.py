from PyQt5 import  QtWidgets, QtCore, QtGui
from gforce import GForceProfile, NotifDataType, DataNotifFlags
import time
import torch
import random
import os
import numpy as np
from  pagewindow import PageWindow
from datawindow import DataWindow
from customcanvas import CustomFigCanvaswoRMS
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from datetime import datetime
from pyqtgraph import PlotWidget, mkPen,PlotCurveItem
from statistics import mode
from modeltraining import loadLogRegr,loadwoOperator, loadwithOperator,get_class
def set_cmd_cb(resp):
    print('Command result: {}'.format(resp))

# try to create a file and write it based on realtime 
# now = datetime.now()
# dt_string = now.strftime("%d/%m/%Y%H:%M:%S")
# os.makedirs(os.path.dirname(f"recordingfiles/{dt_string}.txt"), exist_ok=True)
# file1 = open(f"recordingfiles/{dt_string}.txt","w")
channels = []

actions = list(range(1,4))*2 # Containing numbers 1 to 9 repeats 3 times
random.shuffle(actions)
actionLabelglobal = None
actionImageglobal = None

OFFSET = 121
PEAK = 0
PEAK_MULTIPLIER = 0
ACTION = 0
REP = 0
BASELINE = 100
BASELINE_MULTIPLIER = 100
OFFSET_RMS = 0
#STARTED = False
reg = None
currentaction = "Rest"
ACTIONS = {
    1: ["Rest",             "img/Rest.png",             (None, None),  0],
    2: ["Flexion",          "img/Flexion.png",          (None, None),  0],
    3: ["Extension",        "img/Extension.png",        (None, None),  0],
    4: ["Close palm",       "img/Close.png",            (None, None),  0]
    }

packet_cnt = 0
start_time = 0
FORWARD = 0
ind_channel = 0

# capture incoming EMG data, store it in a global variable channels
def ondata(data):
    #global STARTED
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
    #actions += [ACTION*2+120]*128
    # if STARTED:
    #     file1.write(' '.join(map(str, extracted_data)) +"\n")#+' ' + str(ACTION) +' ' + str(REP) +"\n")

class SearchWindow(PageWindow):
    def __init__(self, GF):
        super().__init__()
        self.initUI()
        self.GF = GF
        self.devices = []

    def initUI(self):
        self.setWindowTitle("Scan for device")
        self.setGeometry(100, 100, 1500, 900)
        self.UiComponents()

    def goToMain(self):
        self.goto("main")

    # Load the new action and unpack other attributes reated to it and update the feature in QT frame
    def loadNewAction(self, newAction):
        global OFFSET_RMS
        try:
            action_name, action_path, (action_baseline, action_peak), action_rep = ACTIONS[newAction]
            self.subj_motion.setText(f"{newAction}")
            self.subj_rep.setText(f"{action_rep}")
            self.actionLabel.setText(action_name)

            pixmap = QtGui.QPixmap(action_path)
            if not pixmap.isNull():
                self.actionImg.setPixmap(pixmap.scaledToWidth(150))
         
            if action_baseline:
                self.e2.setText(f"{action_peak-action_baseline}")
                # self.myFig.update_amp(float(self.e3.text())* (action_peak-action_baseline))
                OFFSET_RMS = action_baseline
        except Exception as e:
            print("Error during loading Action: ", e)
    
    # scan for available devices and display the result on the user interface
    def scan(self):

        scan_results = self.GF.scan(2) #perform a scan for nearby devices and returns infor about them

        # if have scan results the iterate them and create a button which connect to func make_handleBUtton
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

    # this function attemots to connect to a device, handle connection error
    # clear the UI layout of any exsting widgets, provide feedback to user about the connection status
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

    # create handler functions for different buttons in a graphical user  
    def make_handleButton(self, button, *args):
        def handleButton():
            #global reg, ind_channel, ACTION, REP, PEAK, PEAK_MULTIPLIER, OFFSET, STARTED, BASELINE, BASELINE_MULTIPLIER
            global reg, ind_channel, ACTION, REP, PEAK, PEAK_MULTIPLIER, OFFSET, BASELINE, BASELINE_MULTIPLIER
            #global OFFSET_RMS, file1,actionLabelglobal, FORWARD
            global OFFSET_RMS, FORWARD ,actionLabelglobal, actionImageglobal
            
            # IF the button is scan 
            # set text for l1, invoke scan method
            if button == "scan":
                self.l1.setText("Scanning...")
                QtWidgets.qApp.processEvents()
                self.scan()
                self.scanButton.setText("Scan Again")       
            
            elif button == "connectToDevice":

                self.connect(*args)
                QtWidgets.qApp.processEvents() # processes all pending events in the event queue of the application
                # ensure that any pending events such as UI updates or user interface are handled promptly

                # configure parameters for receiving raw EMG data
                GF.setEmgRawDataConfig(sampRate, channelMask, dataLen, resolution, cb=set_cmd_cb, timeout=1000)
                # enable data notification switches: raw EMG data notificatin
                GF.setDataNotifSwitch(DataNotifFlags['DNF_EMG_RAW'], set_cmd_cb, 1000)
                GF.startDataNotification(ondata)

                # update the UI elements 
                self.setWindowTitle("Visualize EMG Data")
                self.layout1.addWidget(self.skipSignalButton)
                self.layout1.addWidget(self.trainModelButton)

                # actionLabelglobal.setText('Select a model')
                # actionLabelglobal.setFont(QtGui.QFont('Arial', 20))
                # actionLabelglobal.setFixedSize(300,30)
                # actionLabelglobal.setAlignment(QtCore.Qt.AlignCenter)
                # self.layout5.addWidget(actionLabelglobal)
                self.modelTrain = QtWidgets.QLabel()
                self.modelTrain.setText('Select a model')
                self.modelTrain.setFont(QtGui.QFont('Arial', 20))
                self.modelTrain.setFixedSize(300,30)
                self.modelTrain.setAlignment(QtCore.Qt.AlignCenter)
                self.layout5.addWidget(self.modelTrain)


                actionLabelglobal.setFont(QtGui.QFont('Arial', 40))
                # self.actionLabel.setText(actionLabelglobal)
                # self.actionLabel.setFont(QtGui.QFont('Arial', 20))
                actionLabelglobal.setAlignment(QtCore.Qt.AlignCenter)

                actionImageglobal.setAlignment(QtCore.Qt.AlignCenter)
                
                # self.actionImg = QtWidgets.QLabel()
                # self.actionImg.setAlignment(QtCore.Qt.AlignCenter)

                self.layout6 =  QtWidgets.QVBoxLayout()
                self.layout6.addWidget(actionLabelglobal)
                self.layout6.addWidget(actionImageglobal)
                # self.layout6.addWidget(self.actionImg)

                self.layout0.addLayout(self.layout5)
                self.layout.addLayout(self.layout3)
                self.layout.addLayout(self.layout6)

    
                QtWidgets.qApp.processEvents()

                # Wait until the length of channel list exceed 128, when it does, it break out of the loop
                while True:
                    if len(channels)>128: 
                        break
                # self.myFig = CustomFigCanvaswoRMS()
                # self.layout.addWidget(self.myFig)
                #Add the callbackfunc to ..

                # function continously sends data and update UI
                # execute the dataSendLoop function, which continously processes data and update UI
                # this sepeartion of tasks into different threads helps maintain a responsive user interface
                # especially when dealing with time-consuming opeartions like data processing.
                myDataLoop = threading.Thread(name = 'myDataLoop', target = dataSendLoop, daemon = True)
                myDataLoop.start()
            
            elif button == "updateMotion":
                try:
                    self.loadNewAction(int(self.subj_motion.text()))
                except Exception as e:
                    print("Error during update motion: ", e)

            # whe the button back to collect is click then disable the train button 
            # ask to choose the model
            elif button=='backToCollect':
                reg = None
                self.trainModelButton.setEnabled(False)
                self.modelTrain.setText('Select a model')
                QtWidgets.qApp.processEvents()

            elif button == "skipSignal":
                FORWARD += 1000
            
            # Depend on the choice for training model then the 
            # reg training model is set or not
            elif button == "loadLogRegr":
                reg = loadLogRegr()
                self.modelTrain.setText("Logistic Regression")
                self.trainModelButton.setEnabled(True)
            
            elif button == "loadwoOperator":
                reg = loadwoOperator()
                self.modelTrain.setText("Without Operator")
                self.trainModelButton.setEnabled(True)
            
            elif button == "loadwithOperator":
                reg = loadwithOperator()
                self.modelTrain.setText("Operator")
                self.trainModelButton.setEnabled(True)

        return handleButton
    
    # receive the data from some source and add to the figure
    # def addData_callbackFunc(self, value):
    #     # print("Add data: " + str(value))
    #     self.myFig.addData(value)

    def UiComponents(self):
        global actionLabelglobal, actionImageglobal
        # initialized as a vertical box layout, arranges widgets vertically
        self.layout = QtWidgets.QVBoxLayout()

        # initialized as horizontal box layout, arrange widget horizontally
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
        #self.loading = QtWidgets.QLabel()
        
        self.layout1.addWidget(self.scanButton)
        self.layout1.addWidget(self.l1)
        #self.layout1.addWidget(self.loading)
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

def getFeatureMatrix(rawDataMatrix, features, window_size, overlap_factor):
    nChannels, nSamples = rawDataMatrix.shape
    stride = int(window_size * overlap_factor)
    num_stride = int(np.ceil(nSamples/(window_size-stride)))
    featMatrix = np.zeros((nChannels*len(features),num_stride))
    # Define a dictionary that maps feature names to functions that calculate those features
    feature_functions = {
        'RMS': lambda x: np.sqrt(np.mean(x ** 2, axis=1)),
        'MAV': lambda x: np.mean(np.abs(x), axis=1),
        'SSC': lambda x: np.mean(((x[:, 1:-1] - x[:, :-2]) * (x[:, 2:] - x[:, 1:-1])) < 0, axis=1).reshape(-1, 1),
        'ZC': lambda x: np.mean((x[:, :-1] * x[:, 1:] < 0) & (np.abs(x[:, :-1] - x[:, 1:]) > 0), axis=1).reshape(-1, 1),
        'WL': lambda x: np.mean(np.abs(x[:, :-1] - x[:, 1:]), axis=1)
    }
    # Loop over the features
    featIndex = 0
    for feature in features:
        if feature in feature_functions:
            featFunc = feature_functions[feature]
            for i in range(num_stride):
                wdwStrtIdx = i*(window_size-stride)
                if i == num_stride:
                    sigWin = rawDataMatrix[:, wdwStrtIdx:nSamples]
                else:
                    sigWin = rawDataMatrix[:, wdwStrtIdx:(wdwStrtIdx+window_size-1)]
               
                featValues = featFunc(sigWin)
                featValues = featValues.flatten() # Flatten featValues before assigning it to featMatrix
                featMatrix[featIndex:featIndex + nChannels, i] = featValues    
            featIndex += nChannels
    return featMatrix
rms_formuula = lambda x: np.sqrt(np.mean(x ** 2, axis=1))

# continously process data and emit signals using a signal-slot mechanism 
def dataSendLoop():
    # Setup the signal-slot mechanism.
    mySrc = Communicate()
    # whenever data is received, the addData_callbackFunc will be invoked
    #mySrc.data_signal.connect(addData_callbackFunc)
    #time.sleep(3)
    # global actionLabelglobal, ACTIONS,FORWARD, reg
    global actionLabelglobal,actionImageglobal, ACTIONS,FORWARD, reg
    while(True):
        #channels[i:i+50*8]
        predictedclasses = []
        for j in range (8):
            
            try:
                datawindow = channels[FORWARD:FORWARD+100*8]
                if datawindow:
                    datastack = np.stack([np.array(datawindow[k::8]) for k in range (8)]).astype('float32') - OFFSET
                    #mean_in_window = datastack.mean(1) # should have size (8,)
                    rms_ = rms_formuula(datastack/255)
                    print(rms_.shape, datastack.shape)
                    rms = rms_.sum()
                    if reg:
                        pred_class = get_class(reg, torch.tensor(rms_.reshape(-1,8))) 
                        predictedclasses.append(pred_class)
                    #     mySrc.data_signal.emit( list(datastack.mean(1)))
                    # else:
                    #     mySrc.data_signal.emit(list(datastack.mean(1)))
                    FORWARD += 50*8 
                if (len(channels) - FORWARD) < -50:
                    time.sleep(47/1000)
                if (len(channels) - FORWARD) > 600:
                    time.sleep(10/1000) 
                else:
                    time.sleep(25/1000)
                

            except Exception as e:
                print("Error during plotting:", type(e),e) 
        if predictedclasses:
            smoothenedclass = mode(predictedclasses)
            actionLabelglobal.setText(f'{ACTIONS[smoothenedclass+1][0]}')
            pixmap = QtGui.QPixmap(ACTIONS[smoothenedclass+1][1])
            actionImageglobal.setPixmap(pixmap.scaledToWidth(150))


if __name__ == "__main__":
    import sys
    sampRate = 1000
    channelMask = 0xFF
    dataLen = 128
    resolution = 8

    GF = GForceProfile()

    app = QtWidgets.QApplication(sys.argv)
    w = SearchWindow(GF)
    actionLabelglobal = QtWidgets.QLabel()
    actionImageglobal = QtWidgets.QLabel()


    w.show()
    sys.exit(app.exec_())