from PyQt5 import  QtWidgets
from gforce import GForceProfile
from  datacollection import DataCollectionhWindow
from window import Window

if __name__ == "__main__":
    import sys
    
    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())