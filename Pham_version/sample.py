# !/usr/bin/python
# -*- coding:utf-8 -*-

from gforce import GForceProfile, NotifDataType, DataNotifFlags
import struct
import sys


from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.pyplot import subplots
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import threading

class CustomMainWindow(QMainWindow):
    def __init__(self):
        super(CustomMainWindow, self).__init__()
        # Define the geometry of the main window
        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle("my first window")
        # Create FRAME_A
        self.FRAME_A = QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QColor(210,210,235,255).name())
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)
        self.label = QLabel("My text")
        self.LAYOUT_A.addWidget(self.label, *(0,0))
        # Place the matplotlib figure
        self.myFig = CustomFigCanvas()
        self.LAYOUT_A.addWidget(self.myFig, *(0,1))
        # Add the callbackfunc to ..
        myDataLoop = threading.Thread(name = 'myDataLoop', target = dataSendLoop, daemon = True, args = (self.addData_callbackFunc,))
        timer = QTimer()
        timer.timeout.connect(self.updateTime)
        timer.start()
        myDataLoop.start()
        self.show()
        return

    def addData_callbackFunc(self, value):
        # print("Add data: " + str(value))
        self.myFig.addData(value)
        return
    
    def updateTime(self):
        self.label.setText(f'{ACTION}')
        return self.label


class CustomFigCanvas(FigureCanvas, TimedAnimation):
    def __init__(self):
        
        # The data
        self.xlim = 200
        self.addedData = []
        self.addedLabel = []

        self.n = np.linspace(0, self.xlim - 1, self.xlim)
        self.y = np.tile(((self.n * 0.0) + 50),(9,1))
        #self.labels = (self.n * 0.0) + 50
        # The window
        self.fig, self.axes  = subplots(nrows=4, ncols=2)
        self.fig.suptitle('Initial action', fontsize=16)
        #[[self.ax1, self.ax2], [self.ax3, self.ax4]] 
        # self.ax1 settings
        #for axis_counter in range (8):
            
            #getattr(self, f'ax{axis_counter}').add_line(getattr(self, f'line{axis_counter}'))
        axis_counter = 1
        for axes_col in self.axes:
            for axes_row in axes_col:
                #axes_row.set_xlabel('time')
                axes_row.set_ylabel(f'{axis_counter}')
                setattr(self, f'line{axis_counter}',Line2D([], [], color='blue'))
                axes_row.add_line(getattr(self,f'line{axis_counter}'))
                axes_row.set_xlim(0, self.xlim - 1)
                axes_row.set_ylim(0, 256)
                axis_counter += 1

      
        #self.line1_tail = Line2D([], [], color='red', linewidth=2)
        #self.line1_head = Line2D([], [], color='red', marker='o', markeredgecolor='r')


        #self.ax1.add_line(self.line1_tail)
        #self.ax1.add_line(self.line1_head)

        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval = 20, blit = True)
        return

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        #lines = [self.line1, self.line2, self.line3, self.line4]#, self.line1_tail]#, self.line1_head]
        for l in range(1,9):
            getattr(self,f'line{l}').set_data([], [])
        return

    def addData(self, value):
        
        self.addedData.append(value[0])
        self.addedLabel.append(value[1])
        return

    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass
        return

    def _draw_frame(self, framedata):
        margin = 2
        while(len(self.addedData) > 0):
            self.y = np.roll(self.y, -1,1)
            #self.labels = np.roll(self.labels, -1)
            self.y[:,-1] = self.addedData[0] + [self.addedLabel[0]]
            #self.y[-1,-1] = self.addedLabel[0]
            del(self.addedData[0])
            del(self.addedLabel[0])
        self.line1.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 0, 0 : self.n.size - margin ])
        self.line2.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 1, 0 : self.n.size - margin ])
        self.line3.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 2, 0 : self.n.size - margin ])
        self.line4.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 3, 0 : self.n.size - margin ])
        self.line5.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 4, 0 : self.n.size - margin ])
        self.line6.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 5, 0 : self.n.size - margin ])
        self.line7.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 6, 0 : self.n.size - margin ])
        self.line8.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 8, 0 : self.n.size - margin ])
        #self.fig.suptitle('Action {}'.format(self.y[-1,-1]), fontsize=16)
        #self.line1_tail.set_data(np.append(self.n[-10:-1 - margin], self.labels[-1 - margin]), np.append(self.labels[-10:-1 - margin], self.labels[-1 - margin]))
        #self.line1_head.set_data(self.n[-1 - margin], self.labels[-1 - margin])
        self._drawn_artists = [self.line1, self.line2, self.line3, self.line4, self.line5, self.line6, self.line7, self.line8]#, self.line1_tail]#, self.line1_head]
        return

''' End Class '''


# You need to setup a signal slot mechanism, to
# send data to your GUI in a thread-safe way.
# Believe me, if you don't do this right, things
# go very very wrong..
class Communicate(QObject):
    data_signal = pyqtSignal(list)

''' End Class '''

def dataSendLoop(addData_callbackFunc):
    # Setup the signal-slot mechanism.
    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)

    i = 0
    while(True):
        mySrc.data_signal.emit([channels[i:i+8],ACTION*10+100]) 
        time.sleep(0.1)
        i += 1

# An example of the callback function
file1 = open("myfile.txt","w")
channels = []
ACTION = 0

def set_cmd_cb(resp):
    print('Command result: {}'.format(resp))


def get_firmware_version_cb(resp, firmware_version):
    print('Command result: {}'.format(resp))
    print('Firmware version: {}'.format(firmware_version))

# An example of the ondata

packet_cnt = 0
start_time = 0

def ondata(data):
    global ACTION
    global channels
    if len(data) > 0:
        #print('[{0}] data.length = {1}, type = {2}'.format(time.time(), len(data), data[0]))

        if data[0] == NotifDataType['NTF_QUAT_FLOAT_DATA'] and len(data) == 17:
            quat_iter = struct.iter_unpack('f', data[1:])
            quaternion = []
            for i in quat_iter:
                quaternion.append(i[0])
            #end for
            print('quaternion:', quaternion)

        elif data[0] == NotifDataType['NTF_EMG_ADC_DATA'] and len(data) == 129:
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
            file1.write(', '.join(map(str, extracted_data)) +', ' + str(ACTION) +"\n")
            #print('\n ------- \n')
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
                
            # end if
        # end if
    # end if


def print2menu():
    
    print('_'*75)
    print('0: Exit')
    print('1: Get Firmware Version')
    print('2: Toggle LED')
    print('3: Toggle Motor')
    print('4: Get Quaternion(press enter to stop)')
    print('5: Set EMG Raw Data Config')
    print('6: Get Raw EMG data(set EMG raw data config first please, press enter to stop)')


if __name__ == '__main__':

    sampRate = 500
    channelMask = 0xFF
    dataLen = 128
    resolution = 8

    while True:
        from time import strftime


        GF = GForceProfile()
        
        print("Scanning devices...")

        # Scan all gforces,return [[num,dev_name,dev_addr,dev_Rssi,dev_connectable],...]
        scan_results = GF.scan(5)

        # Display the first menu
        print('_'*75)


        print('0: exit')
        print(scan_results)
        if scan_results == []:
            print('No bracelet was found')
        else:
            for d in scan_results:
                try:
                    #dev_button = tk.Button(text='{0:<1}: {1:^16} {2:<18} Rssi={3:<3}, connectable:{4:<6}'.format(*d), command=button_handler(d))
                    print('{0:<1}: {1:^16} {2:<18} Rssi={3:<3}, connectable:{4:<6}'.format(*d))
                except:
                    pass
            # end for

        # Handle user actions
        button = int(input('Please select the device you want to connect or exit:'))

        if button == 0:
            break
        else:
            print("Connecting")
            addr = scan_results[button-1][2].upper()
            print(addr)
            GF.connect(addr)
            print("Connected")
            # Display the secord menu
            while True:
                time.sleep(1)
                print2menu()
                button = int(input('Please select a function or exit:'))

                if button == 0:
                    break

                elif button == 1:
                    GF.getControllerFirmwareVersion(get_firmware_version_cb, 1000)

                elif button == 2:
                    GF.setLED(False, set_cmd_cb, 1000)
                    time.sleep(3)
                    GF.setLED(True, set_cmd_cb, 1000)

                elif button == 3:
                    GF.setMotor(True, set_cmd_cb, 1000)
                    time.sleep(3)
                    GF.setMotor(False, set_cmd_cb, 1000)

                elif button == 4:
                    GF.setDataNotifSwitch(DataNotifFlags['DNF_QUATERNION'], set_cmd_cb, 1000)
                    time.sleep(1)
                    GF.startDataNotification(ondata)

                    button = input()
                    print("Stopping...")
                    GF.stopDataNotification()
                    time.sleep(1)
                    GF.setDataNotifSwitch(DataNotifFlags['DNF_OFF'], set_cmd_cb, 1000)

                elif button == 5:
                    sampRate = eval(input('Please enter sample value(max 500, e.g., 500): '))
                    channelMask = eval(input('Please enter channelMask value(e.g., 0xFF): '))
                    dataLen = eval(input('Please enter dataLen value(e.g., 128): '))
                    resolution = eval(input('Please enter resolution value(8 or 12, e.g., 8): '))

                elif button == 6:
                    GF.setEmgRawDataConfig(sampRate, channelMask, dataLen, resolution, cb=set_cmd_cb, timeout=1000)
                    GF.setDataNotifSwitch(DataNotifFlags['DNF_EMG_RAW'], set_cmd_cb, 1000)
                    time.sleep(1)
                    GF.startDataNotification(ondata)
                    app = QApplication(sys.argv)
                    QApplication.setStyle(QStyleFactory.create('Plastique'))
                    myGUI = CustomMainWindow()
                    sys.exit(app.exec_())
                    button = input()
                    print("Stopping...")
                    GF.stopDataNotification()
                    time.sleep(1)
                    GF.setDataNotifSwitch(DataNotifFlags['DNF_OFF'], set_cmd_cb, 1000)
            # end while

            break
            
        # end if
    # end while
# end if
