from PyQt5 import  QtWidgets, QtCore, QtGui
from gforce import  DataNotifFlags
import os
from  pagewindow import PageWindow
from customcanvas import CustomFigCanvas_full, CustomFigCanvas_cue_only, CustomFigCanvaswoRMS, CustomFigCanvas_8channels_only
import threading
from datetime import datetime
from helpers import set_cmd_cb, rms_formuula
import random
from communicate import Communicate
import time
import numpy as np
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import matplotlib.pyplot as plt
import csv
import torch
from data_loader import CustomSignalData, CustomSignalData1
from torch.autograd import Variable
from encoder import Encoder as E
from helpers import get_data, get_all_data, get_shift_data, get_operators, plot_cfs_mat, roll_data

DEVICE = torch.device("cpu")

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y%H:%M:%S")
os.makedirs(os.path.dirname(f"recordingfiles/{dt_string}.txt"), exist_ok=True)
file1 = open(f"recordingfiles/{dt_string}.txt","w")


sampRate = 1000
channelMask = 0xFF
dataLen = 128
resolution = 8
channels = []
actions = list(range(1,10))*2
random.shuffle(actions)

OFFSET = 121
PEAK = 0
PEAK_MULTIPLIER = 0
ACTION = 0
REP = 0
BASELINE = 100
BASELINE_MULTIPLIER = 100
OFFSET_RMS = 0
STARTED = False
reg = None
packet_cnt = 0
start_time = 0
FORWARD = 0
ind_channel = 0
delay_time = []

ACTIONS = {
    1: ["Flexion",          "img/Flexion.png",          (None, None),  0],
    2: ["Extension",        "img/Extension.png",        (None, None),  0],
    3: ["Ulnar Deviation",  "img/UlnarDeviation.png",   (None, None),  0],
    4: ["Radial Deviation", "img/RadialDeviation.png",  (None, None),  0],
    5: ["Supination",       "img/Supination.png",       (None, None),  0],
    6: ["Pronation",        "img/Pronation.png",        (None, None),  0],
    7: ["Open palm",        "img/Open.png",             (None, None),  0],
    8: ["Close palm",       "img/Close.png",            (None, None),  0],
    9: ["Rest",             "img/Rest.png",             (None, None),  0]
    }

def getFeatureMatrix(rawDataMatrix, windowLength, windowOverlap):
    rms = lambda sig: np.sqrt(np.mean(sig**2))
    nChannels,nSamples = rawDataMatrix.shape    
    I = int(np.floor(nSamples/(windowLength-windowOverlap)))
    featMatrix = np.zeros([nChannels, I])
    for channel in range(nChannels):
        for i in range (I):
            wdwStrtIdx=i*(windowLength-windowOverlap)
            sigWin = rawDataMatrix[channel][wdwStrtIdx:(wdwStrtIdx+windowLength-1)] 
            featMatrix[channel, i] = rms(sigWin)
    featMatrixData = np.array(featMatrix)
    return featMatrixData

class FFNN(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(FFNN, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(inputSize, 9, bias=False),
            torch.nn.Sigmoid()
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(9, outputSize, bias=False),
            # torch.nn.Softmax(dim=1)
        )

    def forward(self, x, encoder=None):
        if not encoder:
            encoder = self.encoder
        z = encoder(x)
        class_z = self.classifer(z)

        return class_z

class SearchWindow(PageWindow):
    """
    A class representing a window for scanning and connecting to devices.
    provides a user interface for scanning and connecting devices.
    inherits from the PageWindow class and extends its functionality.

    Attributes:
        GF (GForceProfile): An instance of the GForceProfile class for scanning and connecting to devices.
        devices  (): 

    
    Methods:
        initUI(): Initializes the user interface components and sets up the window.
        goToMain(): Navigates to the main page of the application.
        loadNewAction(newAction): Loads information and images related to a specific action.
        scan(): Initiates the device scanning process and updates the UI with scan results.
        connect(*args): Connects to a device using the provided arguments.
        make_handleButton(button, *args): Generates a button handler function based on the provided button type and arguments.
        addData_callbackFunc(value): Callback function for adding data to the UI.
        UiComponents(): Sets up the user interface components, including buttons and labels.
    """
    def __init__(self, GF):
        """
        Initializes a new instance of the SearchWindow class.

        Args:
            GF (GForceProfile): An instance of the GForceProfile class for scanning
                and connecting to devices.
        """
        super().__init__()
        self.initUI()
        self.GF = GF
        self.devices = []

    def initUI(self):
        """
        Initializes the user interface components and sets up the window.
        """
        self.setWindowTitle("Scan for device")
        self.setGeometry(100, 100, 1500, 900)
        self.UiComponents()

    def goToMain(self):
        """
        Navigates to the main page of the application.
        """
        self.goto("main")

    def loadNewAction(self, newAction):
        """
        Loads information and images related to a specific action. Set them into framework.

        Args:
            newAction (string): name of action to acess data in dictioanry ACTIONS.

        """
        global OFFSET_RMS, ACTIONS
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
                self.myFig.update_amp(float(self.e3.text())* (action_peak-action_baseline))
                OFFSET_RMS = action_baseline
        except Exception as e:
            print("Error during loading Action: ", e)
    
    def scan(self):
        """
        Initiates the device scanning process and 
        updates the UI with scan results: set Text device found, create connect to device button.
        Create a button for device connect, layout2 add this button
        layout0 add layout2
        """
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
        """
        Connects to a device using the provided arguments.

        Args:
            *args: Additional arguments that may be required for connecting to the device.
        """
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
        
    def make_handleButton(self, button, *args):
        """
        Generates a button handler function based on the provided button type and arguments.

        Creates a handler function for various button types in the user interface.
        The handler function performs different actions based on the button type and any additional arguments provided.

        Args:
            button (str): The type of button for which the handler function is being created.
            *args: Additional arguments that may be required for handling certain button types.

        Returns:
            function: A handler function for the specified button type.
            scan button: scan devices.
            connectToDevice: connect devices, set up configuration and data transfer for GF force device, update the fraemwork. 
            calibrate: calibrates the EMG data visualization scale and baseline.
            recordMVC: initiates recording of Maximum Voluntary Contraction (MVC) data.
            pauseMVC: Pause record MVC data and save data.
            startRecord: Initiates the recording process for experimental data.
            loadMotion: Loads a new motion/action for recording.
            stopRecord: Stop adn save the raw EMG data into file.
            updateMotion: LOad new action.
        """
        def handleButton():
            global reg,  ACTION, REP, PEAK, PEAK_MULTIPLIER, OFFSET, STARTED, BASELINE, BASELINE_MULTIPLIER
            global OFFSET_RMS, file1, DEVICE
            global delay_time
            
            if button == "scan":
                """
                Set text scan to the button.
                run method scan 
                """
                self.l1.setText("Scanning...")
                QtWidgets.qApp.processEvents()
                self.scan()
                self.scanButton.setText("Scan Again")       
            
            elif button == "connectToDevice":
                """
                Connect the device 
                Set up configuration and data transfer for GF force device
                set window titlte, layout2 adds layout flo and layout4, layout0 adds layout subj_flo, layout5
                Load first action is first action in dictionary.
                layout adds layout3
                Create myFig (instance of CustomFigCanvas) and adds to layout. 
                Create and execute myDataLoop thread
                """
                self.connect(*args)
                QtWidgets.qApp.processEvents()

                self.GF.setEmgRawDataConfig(sampRate, channelMask, dataLen, resolution, cb=set_cmd_cb, timeout=1000)
                self.GF.setDataNotifSwitch(DataNotifFlags['DNF_EMG_RAW'], set_cmd_cb, 1000)
                self.GF.startDataNotification(ondata)

                self.setWindowTitle("Visualize EMG Data")

                self.layout2.addLayout(self.flo)
                self.layout2.addLayout(self.layout4) 

                self.layout0.addLayout(self.subj_flo)
                self.layout0.addLayout(self.layout5)

                self.loadNewAction(1)
                self.layout.addLayout(self.layout3)
                
                QtWidgets.qApp.processEvents()

                while True:
                    if len(channels)>128: 
                        break
                self.myFig = CustomFigCanvas_cue_only()
                self.layout.addWidget(self.myFig)
                #Add the callbackfunc to ..
                myDataLoop = threading.Thread(name = 'myDataLoop', target = dataSendLoop, daemon = True, args = (self.addData_callbackFunc,))
                myDataLoop.start()

            elif button == "caliberate":
                """
                Update the figure based text vaöue of e1, e3, e2 and BASELINES
                """
                self.myFig.update_scale(int(self.e1.text()))
                self.myFig.update_amp(float(self.e3.text())* (float(self.e2.text())-BASELINE))
                OFFSET_RMS = BASELINE

            elif button == "recordMVC":
                """
                Set the PEAK_MULTIPLIER, BASELINE_MULTIPLIER, and OFFSET_RMS, to prepare for recording
                """
                PEAK_MULTIPLIER = 1
                BASELINE_MULTIPLIER = 1
                OFFSET_RMS = 0
                BASELINE = 10
                self.recordMVCButton.setText("Recording...")
                self.pauseMVCButton.setEnabled(True)
                self.recordMVCButton.setEnabled(False)

            elif button == "pauseMVC":
                """
                Define the current action by the value text of subj_motion.
                Update the the current action in the dictionary with BASELINE, PEAK, 
                Load the next action.
                """
                current_action = int(self.subj_motion.text())
                ACTIONS[current_action][2] = (BASELINE, PEAK)
                ACTIONS[current_action][3] = 1

                #self.e2.setText(f"{PEAK * PEAK_MULTIPLIER}")
                PEAK_MULTIPLIER = 0
                PEAK = 0
                BASELINE_MULTIPLIER = 100

                self.recordMVCButton.setText("Record MVC")
                self.pauseMVCButton.setEnabled(False)
                self.recordMVCButton.setEnabled(True)
                self.loadNewAction( current_action+ 1)

            elif button == "startRecord":
                """
                Update the amplitude of figure by e3*e2
                open the file in folder recordingfiles 
                """
                self.myFig.update_amp(float(self.e3.text())* float(self.e2.text()))
                self.recordSamplButton.setText("Recording ...")
                self.recordSamplButton.setEnabled(False)
                self.loadMotionButton.setEnabled(False)
                file1 = open(f"recordingfiles/{dt_string}.txt","w")
                STARTED= True
                space = [0] * 300
                delay_time.extend(space)

            elif button == "loadMotion":
                """
                Because action is a list of random actions: actions * repetitions 
                load the first element of list actions
                Update the number of actiosn left.
                """
                self.loadNewAction(actions.pop(0))
                self.loadMotionButton.setText(f"Load Random Motion ({len(actions)} left)")
            
            elif button == "stopRecord":
                """
                Open the folder, open the file text based on subject, shift, motion, rep
                Update the current action last element value +1 
                Set adn enable the Record Experiment. 
                """
                STARTED = False
                file1.close()
                os.makedirs(os.path.dirname(f"Subject_{self.subj_name.text()}/Shift_{self.subj_shift.text()}/"), exist_ok=True)
                name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File',
                                                             f"Subject_{self.subj_name.text()}/Shift_{self.subj_shift.text()}/Motion_{self.subj_motion.text()}_Rep_{self.subj_rep.text()}.txt",  
                                                             "Text Files(*.txt)")
                
                try:
                    print(name)
                    os.makedirs(os.path.dirname(name[0]), exist_ok=True)
                    os.system(f'cp {file1.name} {name[0]}')  
                    current_action = int(self.subj_motion.text())
                    ACTIONS[current_action][-1] += 1
                    self.loadMotionButton.setEnabled(True)
                    
                    if (len(channels) - FORWARD) < -50:
                        FORWARD = FORWARD - 200
                    elif (len(channels) - FORWARD) > 600:
                        FORWARD = FORWARD + 400
                    elif (len(channels) - FORWARD) > 800:
                        FORWARD = FORWARD + 600

                except Exception as e:
                    print("Error during saving: ", e)

                self.recordSamplButton.setText("Record Experiment")
                self.recordSamplButton.setEnabled(True)
                
                

            elif button == "updateMotion":
                """
                Load new action.
                """
                try:
                    self.loadNewAction(int(self.subj_motion.text()))
                except Exception as e:
                    print("Error during update motion: ", e)

            elif button=='trainModel':
                self.trainModelButton.setText('Back to Collection Mode')
                self.trainModelButton.clicked.connect(self.make_handleButton("backToCollect"))
                QtWidgets.qApp.processEvents()
                reg = load_NonLinearmodel()

            elif button=='backToCollect':
                
                self.trainModelButton.clicked.connect(self.make_handleButton("trainModel"))
                self.trainModelButton.setText('Train Model')
                QtWidgets.qApp.processEvents()
                reg = None
            elif button == "skipSignal":
                #FORWARD += 1000
                file2 = "recordingfiles/new_timer_10.csv"
                with open(file2, 'w', newline='') as csvfile:
                    # Create a CSV writer object
                    csv_writer = csv.writer(csvfile)
                    
                    # Write each element of the list 'a' as a separate row in the CSV file
                    for item in delay_time:
                        csv_writer.writerow([item])

            elif button == "runModel":
                    subject = self.subj_name.text()
                    No_shift = str(int(self.subj_shift.text()) + 1)

                    Fs = 1000
                    windowLength = int(np.floor(0.1*Fs))  #160ms
                    windowOverlap =  int(np.floor(50/100 * windowLength))

                    X_train = np.zeros([0,8])
                    y_train= np.zeros([0])
                    X_test = np.zeros([0,8])
                    y_test = np.zeros([0])
                    for shift in range(0,int(No_shift)): 
                        for files in sorted(os.listdir(f'Subject_{subject}/Shift_{shift}/')):
                            _, class_,_, rep_ = files.split('_')
                            if int(class_) in [1,2,3,4,5,6,7,8,9]:
                                df = pd.read_csv(f'Subject_{subject}/Shift_{shift}/{files}',skiprows=0,sep=' ',header=None)
                                data_arr = np.stack([np.array(df.T[i::8]).T.flatten().astype('float32') for i in range (8)])
                                data_arr -= 121
                                data_arr /= 255.0
                                feaData = getFeatureMatrix(data_arr, windowLength, windowOverlap)
                                
                                if not class_.startswith('9'):
                                    rms_feature = feaData.sum(0)
                                    baseline = 2*rms_feature[-50:].mean()
                                    start_ = np.argmax(rms_feature[::1]>baseline)
                                    end_  = -np.argmax(rms_feature[::-1]>baseline)
                                    feaData = feaData.T[start_:end_]
                                else:
                                    feaData = feaData.T

                                if rep_.startswith('2'):
                                    X_test, = np.concatenate([X_test,feaData])
                                    y_test = np.concatenate([y_test,np.ones_like(feaData)[:,0]*int(class_)-1])
                                else:
                                    X_train = np.concatenate([X_train,feaData])
                                    y_train= np.concatenate([y_train,np.ones_like(feaData)[:,0]*int(class_)-1])

                    # Data import 
                    all_X_train, all_y_train, all_shift_train = get_all_data(X_train, y_train)
                    all_X_test, all_y_test, all_shift_test = get_all_data(X_test, y_test)

                    all_X1_train, all_X2_train, all_shift_1_train, all_shift_2_train, all_y_shift_train = get_shift_data(all_X_train, all_shift_train, all_y_train)
                    all_X1_test, all_X2_test, all_shift_1_test, all_shift_2_test, all_y_shift_test = get_shift_data(all_X_test, all_shift_test, all_y_test)

                    # Data loader
                    traindataset = CustomSignalData(get_tensor(X_train), get_tensor(y_train))
                    #testdataset = CustomSignalData(get_tensor(X_test), get_tensor(y_test))

                    trainloader = torch.utils.data.DataLoader(traindataset, batch_size = 1, shuffle=True)
                    #testloader = torch.utils.data.DataLoader(testdataset, batch_size=24, shuffle=True)

                    all_train_dataset = CustomSignalData(get_tensor(all_X_train), get_tensor(all_y_train))
                    alltrainloader = torch.utils.data.DataLoader(all_train_dataset, batch_size = 102, shuffle=True)

                    triplet_train_dataset = CustomSignalData1(get_tensor(all_X1_train), get_tensor(all_X2_train), get_tensor(all_shift_1_train), get_tensor(all_shift_2_train), get_tensor(all_y_shift_train))
                    triplettrainloader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size = 102, shuffle=True)

                    # Operator
                    M = torch.diag(torch.ones(8)).roll(-1,1)
                    used_bases = [torch.linalg.matrix_power(M,i).to(DEVICE) for i in range (8)]

                    # Logistic Regresison models 
                    reg = LogisticRegression(penalty='l2', C=100).fit(X_train, y_train)
                    dump(reg, 'LogisticRegression1.joblib')
                    accuracies_LosReg = []
                    for i in range (-4, 4):
                        X_test_shift = roll_data(X_train, i)
                        accuracies_LosReg.append(reg.score(X_test_shift,y_train))                   
                    
                    # Feed Forward Neural Network
                    inputDim = 8     # takes variable 'x' 
                    outputDim = 9      # takes variable 'y'
                    learningRate = 0.005

                    model = FFNN(inputDim, outputDim)
                    model = model.to(DEVICE)
                    
                    crit = torch.nn.CrossEntropyLoss()
                    acc_record = []
                    params_clf = list(model.parameters())# + list(encoder.parameters())
                    optim = torch.optim.Adam(params_clf, lr=learningRate)
                    
                    epochs = 200
                    #encoder = encoder.to(device)
                    for epoch in range(epochs):
                        model.train()

                        # Converting inputs and labels to Variable
                        for inputs, labels, _, _ in alltrainloader:
                            inputs = inputs.to(DEVICE)
                            labels = labels.to(DEVICE)
                            labels = labels.long()
                            labels = labels.flatten()
                            outputs = model(inputs, None)
                            optim.zero_grad()
                            # get loss for the predicted output
                            losss = crit(outputs, labels) #+ 0.001 * model.l1_regula()
                            # get gradients w.r.t to parameters
                            losss.backward()
                            # update parameters
                            optim.step()

                        # if not epoch %20:
                        #     train_acc = clf_acc(model, alltrainloader,encoder= None)
                        #     #test_acc = clf_acc(model, alltestloader, encoder = None)
                        #     acc_record += [train_acc]# [(train_acc, test_acc)]

                    accuracies_ffnn = []
                    for i in range (-4, 4):
                        X_test_shift = roll_data(X_train, i)
                        test_shift_dataset = CustomSignalData(get_tensor(X_test_shift), get_tensor(y_train))
                        testshiftloader = torch.utils.data.DataLoader(test_shift_dataset, batch_size=24, shuffle=True)
                        accuracies_ffnn.append(clf_acc(model, testshiftloader, encoder = None))

                    torch.save(model.state_dict(), "modelwoOperator.pt")

                    # Check the accuracies
                    if accuracies_LosReg and accuracies_ffnn:
                        self.trainButton.setText("Done")
                        print(f'Logistic Regression: {accuracies_LosReg}')
                        print(f'FFNN: {accuracies_ffnn}')
                    else:
                        print("Error:")

        return handleButton
    
    def addData_callbackFunc(self, value):
        """
        add new value to the myFig through method addData.
        """
        self.myFig.addData(value)

    def UiComponents(self):
        """
        Set up user interface components: buttons, labels, fields, layout of window.
        Main Layout(self.layout): 
            Child layout: layout0, layout3, self.myFig
            Contents: layout1("scan" button and label)

        layout0: 
            Child layout: layout1, layout2, layout4, layout5, Subject Form Layout
            Contents: layout1("scan" button and label)

        layout1:
            child widget: "scan" button and label

        layout2: 
            Child layout: layout flo, layout4

        layout3:
            Child Widgets: Input fields and buttons related to calibration and recording

        layout4:
            Child Widgets: Buttons related to recording MVC and calibration (self.recordMVCButton, self.pauseMVCButton, caliberateButton)
        
        layout5:
            Child Widgets: Label and image related to the current action being performed (self.actionLabel, self.actionImg)
        
        Form Layout (self.flo):
            Contents: Input fields for EMG scale, peak, and MVC scale (self.e1, self.e2, self.e3)
        
        Subject Form Layout (self.subj_flo):
            Contents: Input fields for subject name, motion, repetition, and shift (self.subj_name, self.subj_motion, self.subj_rep, self.subj_shift)
        """
        global actions
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
        self.e1 = QtWidgets.QLineEdit("20")
        self.e1.setValidator(QtGui.QIntValidator())
        self.e1.setMaxLength(4)
        self.e1.setAlignment(QtCore.Qt.AlignLeft)
        self.e1.setFixedSize(200, 32)


        self.e2 = QtWidgets.QLineEdit("0")
        self.e2.setValidator(QtGui.QDoubleValidator(0,1,2))
        self.e2.setAlignment(QtCore.Qt.AlignLeft)
        self.e2.setFixedSize(200, 32)

        self.e3 = QtWidgets.QLineEdit("0.3")
        self.e3.setValidator(QtGui.QDoubleValidator(0,1,2))
        self.e3.setAlignment(QtCore.Qt.AlignLeft)
        self.e3.setFixedSize(200, 32)

        self.flo = QtWidgets.QFormLayout()
        
        self.flo.addRow("EMG Scale          ",self.e1)
        self.flo.addRow("Peak               ",self.e2)
        self.flo.addRow("MVC Scale          ",self.e3)

        self.layout4 = QtWidgets.QHBoxLayout()
        
        self.layout4.setAlignment(QtCore.Qt.AlignLeft)
        caliberateButton = QtWidgets.QPushButton("Caliberate")
        caliberateButton.clicked.connect(self.make_handleButton("caliberate"))
        caliberateButton.setFixedSize(120,30)

        self.recordMVCButton = QtWidgets.QPushButton("Record MVC")
        self.recordMVCButton.clicked.connect(self.make_handleButton("recordMVC"))
        self.recordMVCButton.setFixedSize(100,30)

        self.pauseMVCButton = QtWidgets.QPushButton("Pause")
        self.pauseMVCButton.setEnabled(False)
        self.pauseMVCButton.clicked.connect(self.make_handleButton("pauseMVC"))
        self.pauseMVCButton.setFixedSize(100,30)

        self.layout4.addWidget(self.recordMVCButton)
        self.layout4.addWidget(self.pauseMVCButton)
        self.layout4.addWidget(caliberateButton)

        self.loadMotionButton = QtWidgets.QPushButton(f"Load Random Motion ({len(actions)} left)")
        self.loadMotionButton.clicked.connect(self.make_handleButton("loadMotion"))
        self.loadMotionButton.setFixedSize(300,30)

        self.recordSamplButton = QtWidgets.QPushButton("Record Experiment")
        self.recordSamplButton.clicked.connect(self.make_handleButton("startRecord"))
        self.recordSamplButton.setFixedSize(150,30)
        
        stopSamplButton = QtWidgets.QPushButton("Stop")
        stopSamplButton.clicked.connect(self.make_handleButton("stopRecord"))
        stopSamplButton.setFixedSize(150,30)

        self.trainModelButton = QtWidgets.QPushButton("Train model")
        self.trainModelButton.clicked.connect(self.make_handleButton("trainModel"))
        self.trainModelButton.setFixedSize(150,30)
        
        self.skipSignalButton = QtWidgets.QPushButton("Refresh")
        self.skipSignalButton.clicked.connect(self.make_handleButton("skipSignal"))
        self.skipSignalButton.setFixedSize(150,30)

        self.trainButton = QtWidgets.QPushButton("Run model")
        self.trainButton.clicked.connect(self.make_handleButton("runModel"))
        self.trainButton.setFixedSize(150,30)

        self.layout3.addWidget(self.loadMotionButton)
        self.layout3.addWidget(self.recordSamplButton)
        self.layout3.addWidget(stopSamplButton)
        self.layout3.addWidget(self.skipSignalButton)
        self.layout3.addWidget(self.trainButton)

        self.subj_name = QtWidgets.QLineEdit("1")
        self.subj_name.setValidator(QtGui.QIntValidator())
        self.subj_name.setMaxLength(4)
        self.subj_name.setAlignment(QtCore.Qt.AlignLeft)
        self.subj_name.setFixedSize(150, 32)

        self.subj_motion = QtWidgets.QLineEdit("1")
        self.subj_motion.setValidator(QtGui.QDoubleValidator(0,1,2))
        self.subj_motion.textEdited.connect(self.make_handleButton("updateMotion"))
        self.subj_motion.setAlignment(QtCore.Qt.AlignLeft)
        self.subj_motion.setFixedSize(150, 32)

        self.subj_rep = QtWidgets.QLineEdit("1")
        self.subj_rep.setValidator(QtGui.QIntValidator())
        self.subj_rep.setAlignment(QtCore.Qt.AlignLeft)
        self.subj_rep.setFixedSize(150, 32)

        self.subj_shift = QtWidgets.QLineEdit("0")
        self.subj_shift.setValidator(QtGui.QIntValidator())
        self.subj_shift.setAlignment(QtCore.Qt.AlignLeft)
        self.subj_shift.setFixedSize(150, 32)

        self.subj_flo = QtWidgets.QFormLayout()
        
        self.subj_flo.addRow("Subject       ",self.subj_name)
        self.subj_flo.addRow("Motion        ",self.subj_motion)
        self.subj_flo.addRow("Rep           ",self.subj_rep)
        self.subj_flo.addRow("Shift         ",self.subj_shift)

        self.layout5 = QtWidgets.QVBoxLayout()
        self.layout5.setAlignment(QtCore.Qt.AlignCenter)
        self.layout5.setContentsMargins(0, 0, 0, 0)

        self.actionLabel = QtWidgets.QLabel()
        self.actionLabel.setFont(QtGui.QFont('Arial', 20))
        self.actionLabel.setFixedSize(300,30)
        self.actionLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.actionImg = QtWidgets.QLabel()
        self.actionImg.setAlignment(QtCore.Qt.AlignCenter)
        self.layout5.addWidget(self.actionLabel)
        self.layout5.addWidget(self.actionImg)
        

def ondata(data):
    """
    Function to write data into a gloabl file: file1 

    Args:
    data (array 2 dimens/ Pandan dataframe): The raw data.
    
    """
    global STARTED, channels, file1
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

    if STARTED:
        file1.write(' '.join(map(str, extracted_data)) +"\n")

def dataSendLoop(addData_callbackFunc):
    # Setup the signal-slot mechanism.
    """
    Continuous loop for sending data to the callback function for plotting.

    Args:
        addData_callbackFunc (function): Callback function to which the data is sent.

    This function sets up the signal-slot mechanism and continuously sends data to the specified callback function for plotting.
    It iterates over the data channels, calculates features, and emits the data to the callback function.

    Note:
        This function assumes the availability of global variables: PEAK, PEAK_MULTIPLIER, BASELINE, OFFSET_RMS, BASELINE_MULTIPLIER,
        ACTIONS, FORWARD, and reg.

    """
    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)
    global PEAK, PEAK_MULTIPLIER, BASELINE, OFFSET_RMS, BASELINE_MULTIPLIER,ACTIONS,FORWARD, delay_time

    while(True):
        #channels[i:i+50*8]
        #for j in range (8):
        try:
            datawindow = channels[FORWARD:FORWARD+100*8]
            if datawindow:
                datastack = np.stack([np.array(datawindow[k::8]) for k in range (8)]).astype('float32') - OFFSET
                mean_in_window = datastack.mean(1) # should have size (8,)
                rms_ = rms_formuula(datastack/255)
                rms = rms_.sum()- OFFSET_RMS
                
                if OFFSET_RMS:
                    mySrc.data_signal.emit([rms])
                else:
                    BASELINE = min(rms*BASELINE_MULTIPLIER, BASELINE)
                    PEAK = max(rms*PEAK_MULTIPLIER, PEAK)
                    mySrc.data_signal.emit([rms])
                FORWARD += 50*8
            if (len(channels) - FORWARD) < -50:
                time.sleep(47/1000)
            if (len(channels) - FORWARD) > 600:
                time.sleep(10/1000) 
            else:
                time.sleep(25/1000)
            delay_time.append(len(channels) - FORWARD)
            print(f'Current cursor: {FORWARD} - Collected data: {len(channels)} delay: {(len(channels) - FORWARD)}')

        except Exception as e:
            print("Error during plotting:", type(e),e) 

def get_tensor(arr):
    return torch.tensor(arr, device=DEVICE,dtype=torch.float )

def rotate_batch(x, d, out_features):
    rotated = torch.empty(x.shape, device=DEVICE)
    for i in range (x.shape[0]):
        rotated[i] = used_bases[d[i]].matmul(x[i]) 
    return rotated

def clf_acc(model, loader, masks = None, encoder = None):
    model.eval()
    correct = 0
    iter = 0
    with torch.no_grad():
        for inputs, labels,_,_ in loader:
            inputs = inputs.to(DEVICE)
            if masks is not None:
                inputs = inputs * masks[:inputs.size()[0]]
            labels = labels.to(DEVICE)
            labels = labels.flatten()
            if encoder:
                pred = model(inputs, encoder)
            else:
                pred = model(inputs)
            correct += (1-torch.abs(torch.sign(torch.argmax(pred,dim = 1)- labels))).mean().item()
            iter += 1
    return correct/iter

def compute_accuracy(a, b, loader):
    a.eval()
    b.eval()
    
    correct = 0
    iter = 0
    
    with torch.no_grad():
        for inputs1, inputs2, shift1, shift2, labels, _ in loader:
            inputs1 = inputs1.to(DEVICE)
            inputs2 = inputs2.to(DEVICE)
            shift1 = -shift1.int().flatten().to(DEVICE)
            shift2 = -shift2.int().flatten().to(DEVICE)
            labels = labels.flatten().to(DEVICE)
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            y1 = a(inputs1)
            y_tr_est1 = rotate_batch(y1,shift1,6)
            y_tr1 = b(y_tr_est1)

            y2 = a(inputs2)
            y_tr_est2 = rotate_batch(y2,shift1,6)
            y_tr2 = b(y_tr_est2)

            correct += (1-torch.abs(torch.sign(torch.argmax(y_tr1,dim = 1)- labels))).mean().item() + \
                    (1-torch.abs(torch.sign(torch.argmax(y_tr2,dim = 1)- labels))).mean().item()
            iter += 1
    return correct * 0.5 / iter