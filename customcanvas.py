import numpy as np
import queue
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.pyplot import subplots
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from  concurrent.futures import ThreadPoolExecutor

# so this module creates 2 classes with similar attributes and methods with the purpose to plot in real time data in 2 subplot. 
# the first subplot has the 8 channels graphs. The second subplot has the single plot whihc is the rms avarage of 8 channels


# the CustomFigCanvas class set up a Matplot figure with 2 subplots, initialize lines for each channels
# and provides for updating data and visualizing real-time animated plots. The class is designed to handle 
# and display data for multiple channels over time. 

class CustomFigCanvas_full(FigureCanvas, TimedAnimation):
    def __init__(self):
        
        # The data
        self.scale = 20
        self.xlim = 200
        self.amplitude = 0.5
        self.addedData = queue.Queue()
        self.addedLabel = queue.Queue()
        self.timeline = np.arange(0,25,5)-25
        self.timelinex = np.arange(0,500,100)
        self.n = np.linspace(0, 299, 500)
        self.convolemask = np.ones(15)/15
        self.cue_line = np.zeros(500)#, np.linspace(0, self.amplitude, 100), np.ones(200) * self.amplitude,np.linspace( self.amplitude, 0,100),np.zeros(150) ])
        self.extra = np.arange(0,self.scale*9,self.scale)
        print(self.cue_line.shape)
        self.y = np.zeros([500,9]) - self.extra
        self.show_both_subplots = False
        #self.labels = (self.n * 0.0) + 50
        
        # The window
        self.fig, self.axes  = subplots(1,2, figsize=(8, 5))#, gridspec_kw={'height_ratios': [1, 1]})

        for i in range(1):

            setattr(self, f'line{i}',Line2D([], [])) 
            self.axes[1].add_line(getattr(self,f'line{i}')) 
        for i in range(1,9):

            setattr(self, f'line{i}',Line2D([], [])) 
            self.axes[0].add_line(getattr(self,f'line{i}')) 
        
        self.line9 = Line2D([], [], color='red', alpha=0.3)
        self.line10 = Line2D([], [], color='red', marker='o', markersize=10)
        self.axes[1].add_line(self.line9) 
        self.axes[1].add_line(self.line10) 
        self.labels = [f'Channel {chan}' for chan in range (1,9)]
        self.axes[0].set_xlim(0, 500)
        
        self.axes[0].set_ylim(-self.scale*9, -0)
        self.axes[0].set_yticks(-self.extra[1:])
        self.axes[0].set_yticklabels(self.labels)

        self.axes[1].set_xlim(0, 500)
        self.axes[1].set_ylim(0,1.5)
        self.axes[1].set_xticks(self.timelinex)
        self.axes[1].set_xticklabels(self.timeline)
        self.axes[1].set_yticks([])
        self.axes[1].set_yticks([], minor=True)
        
        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval = 50,repeat=True, blit = True)
        

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        #lines = [self.line1, self.line2, self.line3, self.line4]#, self.line1_tail]#, self.line1_head]
        for l in range(11):
            getattr(self,f'line{l}').set_data([], [])
        #[self.axes.plot(self.extra[i-1])[0] for i in range(1,9)]

    def addData(self, value):
        self.addedData.put(value)
        #self.addedLabel.put(value[1])
        
    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass
    def update_amp(self, new_val):
        self.amplitude = new_val
        self.cue_line = np.hstack([np.zeros(440), np.linspace(0, new_val, 40), np.ones(200) * new_val,np.linspace( new_val, 0,40)  ]) # 1sec = 20
        
    def update_scale(self, new_val):
        return
        #self.scale = new_val
        #self.extra = np.arange(0,new_val*9,new_val)
        #self.axes[0].set_ylim(-self.scale*9, -0)
        #self.axes[0].set_yticks(-self.extra[1:])

    def set_line(self, idx):
        getattr(self, f"line{idx}").set_data( self.n, range(2000))
    def moving_average(self, a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    def _draw_frame(self, framedata):
        try:
            new_data = self.addedData.get_nowait()
            self.y = np.roll(self.y,-1,0)
            self.cue_line = np.roll(self.cue_line,-1)
            self.y[-1,:] = new_data
            self.y[-1,:] -= self.extra
            plottingdata = self.moving_average(self.y[:,0], 15)
            print(plottingdata.shape)
            #plottingdata = np.concatenate([self.y,np.zeros([100])])

            #self.n = np.roll(self.n, -1)
            #self.n[-1] = self.n[-2] + 1
            
        #self.labels = np.roll(self.labels, -1)
        except Exception as e:
            print("Error:", type(e),e)
        
       
                #p.map(self.set_line, range(1,9))
        try:
            self.line0.set_data(range(400),plottingdata[86:])
            for i in range (1,9): 
            #return [self.axes.plot(self.y[i-1,:])[0] for i in range(1,9)]#
                getattr(self, f"line{i}").set_data(range(500),self.y[:,i])
            self.line9.set_data(range(500), self.cue_line[:500])
            self.line10.set_data(400,plottingdata[-1])
            self.axes[1].set_ylim(-0.01, self.amplitude * 2)
            self.axes[1].set_yticks([], minor=True)
            self._drawn_artists = [getattr(self, f"line{i}") for i in range(11)]#, self.line2, self.line3, self.line4, self.line5, self.line6, self.line7, self.line8]#, self.line1_tail]#, self.line1_head]
        except Exception as e:
            print("Error after get data",e)

        def toggle_subplots(self):
            if self.show_both_subplots:
                # If currently showing both subplots, switch to showing only the second subplot
                self.axes[0].set_visible(False)
                self.show_both_subplots = False
            else:
                # If currently showing only the second subplot, switch back to showing both subplots
                self.axes[0].set_visible(True)
                self.show_both_subplots = True

            # Redraw the canvas to reflect the changes
            self.fig.canvas.draw()
class CustomFigCanvas_cue_only(FigureCanvas, TimedAnimation):
    def __init__(self):
        
        # The data
        self.scale = 20 # Scaling factor for y-axis
        self.xlim = 200 # LImit ont x-axis
        self.amplitude = 0.5 #Initial amplitude value 
        self.addedData = queue.Queue() # Queue for storing incoming data
        self.addedLabel = queue.Queue() # Queue for storing labels
        self.timeline = np.arange(0,25,5)-25 # A timeline array with values for x-axis ticks
        self.timelinex = np.arange(0,500,100) # Array for x-axis tick positions 
        self.n = np.linspace(0, 299, 500) # An array reprenting the x-axis values
        self.convolemask = np.ones(15)/15
        self.cue_line = np.zeros(500)#, np.linspace(0, self.amplitude, 100), np.ones(200) * self.amplitude,np.linspace( self.amplitude, 0,100),np.zeros(150) ])
        self.extra = np.arange(0,self.scale*9,self.scale)
        print(self.cue_line.shape)
        self.y = np.zeros([500,9]) - self.extra # An array reprenting the y-axis ticks
        #self.labels = (self.n * 0.0) + 50
        
        # The window
        self.fig, self.axes  = subplots(figsize=(8, 5))#, gridspec_kw={'height_ratios': [1, 1]})

        # 2 subplots
        for i in range(1):

            setattr(self, f'line{i}',Line2D([], [])) 
            # self.axes[1].add_line(getattr(self,f'line{i}')) 
            self.axes.add_line(getattr(self,f'line{i}')) 
        # for i in range(1,9):

        #     setattr(self, f'line{i}',Line2D([], [])) 
        #     self.axes[0].add_line(getattr(self,f'line{i}')) 
        
        # self.line9 = Line2D([], [], color='red', alpha=0.3)
        # self.line10 = Line2D([], [], color='red', marker='o', markersize=10)
        self.line1 = Line2D([], [], color='red', alpha=0.3)
        self.line2 = Line2D([], [], color='red', marker='o', markersize=10)
        # self.axes[1].add_line(self.line9) 
        # self.axes[1].add_line(self.line10)

        self.axes.add_line(self.line1) 
        self.axes.add_line(self.line2)  

        # Axis and label setting
        # self.labels = [f'Channel {chan}' for chan in range (1,9)]
        # self.axes[0].set_xlim(0, 500)
        
        # self.axes[0].set_ylim(-self.scale*9, -0)
        # self.axes[0].set_yticks(-self.extra[1:])
        # self.axes[0].set_yticklabels(self.labels)

        # self.axes[1].set_xlim(0, 500)
        # self.axes[1].set_ylim(0,1.5)
        # self.axes[1].set_xticks(self.timelinex)
        # self.axes[1].set_xticklabels(self.timeline)
        # self.axes[1].set_yticks([])
        # self.axes[1].set_yticks([], minor=True)

        self.axes.set_xlim(0, 500)
        self.axes.set_ylim(0,1.5)
        self.axes.set_xticks(self.timelinex)
        self.axes.set_xticklabels(self.timeline)
        self.axes.set_yticks([])
        self.axes.set_yticks([], minor=True)
        
        # FigureCanvas and TimedAnimation Initialization
        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval = 50,repeat=True, blit = True)
        
    # Return an iterator for the frame sequence 
    def new_frame_seq(self):
        return iter(range(self.n.size))

    # Initializing the drawing by setting Line2D objects to empty data
    def _init_draw(self):
        #lines = [self.line1, self.line2, self.line3, self.line4]#, self.line1_tail]#, self.line1_head]
        # for l in range(11):
        for l in range(3):
            getattr(self,f'line{l}').set_data([], [])
        #[self.axes.plot(self.extra[i-1])[0] for i in range(1,9)]

    #Add new data to the addedData
    def addData(self, value):
        self.addedData.put(value)
        #self.addedLabel.put(value[1])
        
    # Extends the _step() method for the TimedAnimation class, handling exceptions and stopping animation if an error occurs.
    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass

    #Updates the amplitude value
    def update_amp(self, new_val):
        self.amplitude = new_val
        self.cue_line = np.hstack([np.zeros(440), np.linspace(0, new_val, 40), np.ones(200) * new_val,np.linspace( new_val, 0,40)  ]) # 1sec = 20
        
    def update_scale(self, new_val):
        return
        #self.scale = new_val
        #self.extra = np.arange(0,new_val*9,new_val)
        #self.axes[0].set_ylim(-self.scale*9, -0)
        #self.axes[0].set_yticks(-self.extra[1:])

    # Sets data for a specific Line2D object
    def set_line(self, idx):
        getattr(self, f"line{idx}").set_data( self.n, range(2000))
    def moving_average(self, a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    # Draws frames by updating the data for each channel.
    def _draw_frame(self, framedata):
        try:
            new_data = self.addedData.get_nowait()
            self.y = np.roll(self.y,-1,0)
            self.cue_line = np.roll(self.cue_line,-1)
            self.y[-1,:] = new_data
            self.y[-1,:] -= self.extra
            plottingdata = self.moving_average(self.y[:,0], 15)
            print(plottingdata.shape)
            #plottingdata = np.concatenate([self.y,np.zeros([100])])

            #self.n = np.roll(self.n, -1)
            #self.n[-1] = self.n[-2] + 1
            
        #self.labels = np.roll(self.labels, -1)
        except Exception as e:
            print("Error:", type(e),e)
        
       
                #p.map(self.set_line, range(1,9))
        try:
            self.line0.set_data(range(400),plottingdata[86:])
            # for i in range (1,9): 
            # #return [self.axes.plot(self.y[i-1,:])[0] for i in range(1,9)]#
            #     getattr(self, f"line{i}").set_data(range(500),self.y[:,i])
            self.line1.set_data(range(500), self.cue_line[:500])
            self.line2.set_data(400,plottingdata[-1])
            # self.axes[1].set_ylim(-0.01, self.amplitude * 2)
            # self.axes[1].set_yticks([], minor=True)
            # self._drawn_artists = [getattr(self, f"line{i}") for i in range(11)]#, self.line2, self.line3, self.line4, self.line5, self.line6, self.line7, self.line8]#, self.line1_tail]#, self.line1_head]
        
            self.axes.set_ylim(-0.01, self.amplitude * 2)
            self.axes.set_yticks([], minor=True)
            self._drawn_artists = [getattr(self, f"line{i}") for i in range(3)]

        except Exception as e:
            print("Error after get data",e)

class CustomFigCanvas_8channels_only(FigureCanvas, TimedAnimation):
    def __init__(self):
        
        # The data
        self.scale = 20 # Scaling factor for y-axis
        self.xlim = 200 # LImit ont x-axis
        self.amplitude = 0.5 #Initial amplitude value 
        self.addedData = queue.Queue() # Queue for storing incoming data
        self.addedLabel = queue.Queue() # Queue for storing labels
        self.timeline = np.arange(0,25,5)-25 # A timeline array with values for x-axis ticks
        self.timelinex = np.arange(0,500,100) # Array for x-axis tick positions 
        self.n = np.linspace(0, 299, 500) # An array reprenting the x-axis values
        self.convolemask = np.ones(15)/15
        self.cue_line = np.zeros(500)#, np.linspace(0, self.amplitude, 100), np.ones(200) * self.amplitude,np.linspace( self.amplitude, 0,100),np.zeros(150) ])
        self.extra = np.arange(0,self.scale*9,self.scale)
        print(self.cue_line.shape)
        self.y = np.zeros([500,9]) - self.extra # An array reprenting the y-axis ticks
        #self.labels = (self.n * 0.0) + 50
        
        # The window
        self.fig, self.axes  = subplots(figsize=(8, 5))#, gridspec_kw={'height_ratios': [1, 1]})

        # 2 subplots
        # for i in range(1):

        #     setattr(self, f'line{i}',Line2D([], [])) 
        #     # self.axes[1].add_line(getattr(self,f'line{i}')) 
        #     self.axes.add_line(getattr(self,f'line{i}')) 
        for i in range(1,9):

            setattr(self, f'line{i}',Line2D([], [])) 
            self.axes.add_line(getattr(self,f'line{i}')) 
        
        # self.line9 = Line2D([], [], color='red', alpha=0.3)
        # self.line10 = Line2D([], [], color='red', marker='o', markersize=10)
        # self.line1 = Line2D([], [], color='red', alpha=0.3)
        # self.line2 = Line2D([], [], color='red', marker='o', markersize=10)
        # self.axes[1].add_line(self.line9) 
        # self.axes[1].add_line(self.line10)

        # self.axes.add_line(self.line1) 
        # self.axes.add_line(self.line2)  

        # Axis and label setting
        self.labels = [f'Channel {chan}' for chan in range (1,9)]
        self.axes.set_xlim(0, 500)
        
        self.axes.set_ylim(-self.scale*9, -0)
        self.axes.set_yticks(-self.extra[1:])
        self.axes.set_yticklabels(self.labels)

        # self.axes[1].set_xlim(0, 500)
        # self.axes[1].set_ylim(0,1.5)
        # self.axes[1].set_xticks(self.timelinex)
        # self.axes[1].set_xticklabels(self.timeline)
        # self.axes[1].set_yticks([])
        # self.axes[1].set_yticks([], minor=True)

        
        
        # FigureCanvas and TimedAnimation Initialization
        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval = 50,repeat=True, blit = True)
        
    # Return an iterator for the frame sequence 
    def new_frame_seq(self):
        return iter(range(self.n.size))

    # Initializing the drawing by setting Line2D objects to empty data
    def _init_draw(self):
        #lines = [self.line1, self.line2, self.line3, self.line4]#, self.line1_tail]#, self.line1_head]
        # for l in range(11):
        for l in range(1,9):
            getattr(self,f'line{l}').set_data([], [])
        #[self.axes.plot(self.extra[i-1])[0] for i in range(1,9)]

    #Add new data to the addedData
    def addData(self, value):
        self.addedData.put(value)
        #self.addedLabel.put(value[1])
        
    # Extends the _step() method for the TimedAnimation class, handling exceptions and stopping animation if an error occurs.
    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass

    #Updates the amplitude value
    def update_amp(self, new_val):
        self.amplitude = new_val
        self.cue_line = np.hstack([np.zeros(440), np.linspace(0, new_val, 40), np.ones(200) * new_val,np.linspace( new_val, 0,40)  ]) # 1sec = 20
        
    def update_scale(self, new_val):
        return
        #self.scale = new_val
        #self.extra = np.arange(0,new_val*9,new_val)
        #self.axes[0].set_ylim(-self.scale*9, -0)
        #self.axes[0].set_yticks(-self.extra[1:])

    # Sets data for a specific Line2D object
    def set_line(self, idx):
        getattr(self, f"line{idx}").set_data( self.n, range(2000))
    def moving_average(self, a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    # Draws frames by updating the data for each channel.
    def _draw_frame(self, framedata):
        try:
            new_data = self.addedData.get_nowait()
            self.y = np.roll(self.y,-1,0)
            self.cue_line = np.roll(self.cue_line,-1)
            self.y[-1,:] = new_data
            self.y[-1,:] -= self.extra
            plottingdata = self.moving_average(self.y[:,0], 15)
            print(plottingdata.shape)
            #plottingdata = np.concatenate([self.y,np.zeros([100])])

            #self.n = np.roll(self.n, -1)
            #self.n[-1] = self.n[-2] + 1
            
        #self.labels = np.roll(self.labels, -1)
        except Exception as e:
            print("Error:", type(e),e)
        
       
                #p.map(self.set_line, range(1,9))
        try:
            
            for i in range (1,9): 
            #return [self.axes.plot(self.y[i-1,:])[0] for i in range(1,9)]#
                getattr(self, f"line{i}").set_data(range(500),self.y[:,i])
            # self.line1.set_data(range(500), self.cue_line[:500])
            # self.line2.set_data(400,plottingdata[-1])
            # self.axes[1].set_ylim(-0.01, self.amplitude * 2)
            # self.axes[1].set_yticks([], minor=True)
            # self._drawn_artists = [getattr(self, f"line{i}") for i in range(11)]#, self.line2, self.line3, self.line4, self.line5, self.line6, self.line7, self.line8]#, self.line1_tail]#, self.line1_head]
        
            self._drawn_artists = [getattr(self, f"line{i}") for i in range(1,9)]

        except Exception as e:
            print("Error after get data",e)

class CustomFigCanvaswoRMS(FigureCanvas, TimedAnimation):
    def __init__(self):
        
        # The data
        self.scale = 20
        self.xlim = 200
        self.amplitude = 0.5
        self.addedData = queue.Queue()
        self.addedLabel = queue.Queue()
        self.timeline = np.arange(0,25,5)-25
        self.timelinex = np.arange(0,500,100)
        self.n = np.linspace(0, 299, 500)
        self.extra = np.arange(self.scale,self.scale*9,self.scale)
        self.y = np.zeros([500,8]) - self.extra

        # The window
        self.fig, self.axes  = subplots(figsize=(100,100))

        for i in range(1,9):

            setattr(self, f'line{i}',Line2D([], [])) 
            self.axes.add_line(getattr(self,f'line{i}')) 
        
        self.labels = [f'Channel {chan}' for chan in range (1,9)]
        self.axes.set_xlim(0, 500)
        
        self.axes.set_ylim(-self.scale*9, -0)
        self.axes.set_yticks(-self.extra)
        self.axes.set_yticklabels(self.labels)
        
        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval = 50,repeat=True, blit = True)
        

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        for l in range(1,9):
            getattr(self,f'line{l}').set_data([], [])

    def addData(self, value):
        self.addedData.put(value)
        #self.addedLabel.put(value[1])
        
    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass
    def update_amp(self, new_val):
        self.amplitude = new_val
        self.cue_line = np.hstack([np.zeros(540), np.linspace(0, new_val, 40), np.ones(200) * new_val,np.linspace( new_val, 0,40)  ]) # 1sec = 20
        
    def update_scale(self, new_val):
        return

    def set_line(self, idx):
        getattr(self, f"line{idx}").set_data( self.n, range(2000))
    def _draw_frame(self, framedata):
        try:
            new_data = self.addedData.get_nowait()
            self.y = np.roll(self.y,-1,0)
            self.y[-1,:] = new_data
            self.y[-1,:] -= self.extra
            
        except Exception as e:
            pass
            #print("Error before plotting ",e)

        try:
            for i in range (1,9): 
                getattr(self, f"line{i}").set_data(range(500),self.y[:,i-1])
            self._drawn_artists = [getattr(self, f"line{i}") for i in range(1,9)]#, self.line2, self.line3, self.line4, self.line5, self.line6, self.line7, self.line8]#, self.line1_tail]#, self.line1_head]
        except Exception as e:
            print("Error after get data ",e)