"""
    * @file: getPoints.py
    * @version: v1.0.0
    * @author: Zhiyu Zhang
    * @desc: get points
    * @date: 2023-07-25 12:56:43
    * @Email: erbiaoger@gmail.com
    * @url: erbiaoger.site

"""

from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QToolTip, QMessageBox,
                             QMainWindow, QHBoxLayout, QVBoxLayout, QFileDialog, QSizePolicy,
                             QSlider, QLabel, QLineEdit, QGridLayout, QGroupBox, QListWidget,
                             QTabWidget)
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt, QPointF

import sys
import matplotlib; matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import os
import pathlib
import h5py
from tqdm import tqdm


class GetPoints(QMainWindow):
    def __init__(self):
        super().__init__()
        self.points = []
        self.curve = []
        self.initUI()

    def initUI(self):
        # 创建一个主窗口的中心部件
        central_widget = QWidget()
        central_widget.setLayout(QHBoxLayout())
        self.setCentralWidget(central_widget)

        self.layout = central_widget.layout()

        # 设置主窗口的大小策略
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setSizePolicy(size_policy)


        self.setMouseTracking(True)
        self.initButtons()
        self.sliderColor()
        self.initFigure()
        self.initfolder()
        self.show()



    def initFigure(self):

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        toolbar = NavigationToolbar(self.canvas, self)

        widFig = QWidget()
        newLayout = QVBoxLayout()
        widFig.setLayout(newLayout)
        newLayout.addWidget(self.canvas, 0)
        newLayout.addWidget(toolbar, 0)
        self.layout.addWidget(widFig, 3)

    def initButtons(self):
        widButtons = QWidget()
        layButtons = QVBoxLayout()
        widButtons.setLayout(layButtons)



        btnOpenFolder = QPushButton("Open Folder")
        btnOpenFolder.clicked.connect(self.openFolder)
        layButtons.addWidget(btnOpenFolder, 0)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        layButtons.addWidget(self.slider, 0)

        btnPoints = QPushButton("Get Points")
        btnPoints.clicked.connect(self.getPoints)
        layButtons.addWidget(btnPoints, 0)

        btnCurve = QPushButton("Draw Curve")
        btnCurve.clicked.connect(self.drawCurve)
        layButtons.addWidget(btnCurve, 0)

        btnNext = QPushButton("Next")
        btnNext.clicked.connect(self.next)
        btnNext.setShortcut('d')
        btnNext.setToolTip('Next (d)')
        layButtons.addWidget(btnNext, 0)
        
        btnMoveData = QPushButton('Get Data', self)
        btnMoveData.clicked.connect(self.moveData)
        btnMoveData.setShortcut('w')
        btnMoveData.setToolTip('Get Data (w)')
        layButtons.addWidget(btnMoveData, 0)

        self.layout.addWidget(widButtons, 0)

    def getPoints(self):
        self.points = []

        self.fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click(event, self.ax))
        

    
    def on_click(self, event, ax):
        
        x = np.argmin(np.abs(self.x - event.xdata))
        y = np.argmin(np.abs(self.t - event.ydata))
        self.points.append([x, y])
        ax.scatter(event.xdata, event.ydata)
        self.canvas.draw()
        if len(self.points) % 2 == 0:
            self.drawCurve()

    def drawCurve(self):
        curve = np.array(self.points)

        x1 = curve[0, 0]
        y1 = curve[0, 1]
        x2 = curve[1, 0]
        y2 = curve[1, 1]
        x = np.arange(x1, x2)

        fx = (y2-y1)/(x2-x1) * (x-x1) + y1
        fx = np.array(list(map(int, fx)))


        self.ax.scatter(self.x[x], self.t[fx])
        self.canvas.draw()

        vel = (self.x[x[-1]] - self.x[x[0]])/(self.t[fx[-1]] - self.t[fx[0]]) * 3.6
        print(vel)
        # if np.abs(vel) > 12:
        data = self.data.copy()
        #data[x, fx] = 1
        for i, j in zip(x, fx):
            data[j-200:j+200, i] *= 100

        name = self.files[self.ID].split('.')[0]
        np.savez(f'{name}.npz', raw=self.data, label=data)

    def sliderColor(self):
        """
        Set speed of animation
        """
        self.slider.setMinimum(1)
        self.slider.setMaximum(500)
        self.slider.valueChanged.connect(self.sliderValueChanged)

    def sliderValueChanged(self, value):
        """
        Set speed of animation, help function of sliderSpeed
        """

        self.imshowData()
        print(value)


    def initfolder(self):
        # Folder
        self.list_widget = QListWidget(self)
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        widFolder = QGroupBox("Folder", self)
        LayoutFolder = QVBoxLayout()
        LayoutFolder.addWidget(self.list_widget, 1)
        widFolder.setLayout(LayoutFolder)
        self.layout.addWidget(widFolder, 0)

    def on_item_clicked(self, item):
        # 获取用户点击的文件项，并打印文件名
        print("文件名:", item.text())
    
        for index, value in enumerate(self.files):
            if value == item.text():
                self.ID = index
                break
        
        self.fpath = os.path.join(self.folderName, item.text())
        
        self.readData(self.fpath)
        self.imshowData()

    def openFolder(self):
        folderName = QFileDialog.getExistingDirectory(self, "Select Directory", "./")
        self.folderName = folderName
        self.files = sorted(os.listdir(folderName))
        self.list_widget.clear()
        for file in self.files:
            self.list_widget.addItem(file)


    def readData(self, filename='./Data/SR_2023-07-20_09-09-38_UTC.h5'):
        self.fname = filename
        fileType = os.path.splitext(filename)[-1]
        if fileType == '.h5':
            # with h5py.File(filename, 'r') as f:
            #     StrainRate = f['/fa1-22070070/Source1/Zone1/StrainRate'][:]
            #     spacing = f['/fa1-22070070/Source1/Zone1'].attrs['Spacing']

            with h5py.File(filename, 'r') as f:
                for a in f:
                    StrainRate = f[f'/{a}/Source1/Zone1/StrainRate'][:]
                    spacing = f[f'/{a}/Source1/Zone1'].attrs['Spacing']


            #self.data = StrainRate.reshape(-1, StrainRate.shape[-1])
            self.dx = spacing[0]
            self.dt = spacing[1] * 1e-3
            nb, nt, nx = StrainRate.shape

            self.data = StrainRate[:, nt//2:, :].reshape(-1, nx)

            self.nt, self.nx = self.data.shape

            self.pre_data = self.data.copy()
            self.vmin = np.nanmin(self.data)
            self.vmax = np.nanmax(self.data)
            
            filepath = pathlib.Path(filename)

        elif fileType == '.dat':
            #filename = 'guangu/2023-07-20-18-40-58-out.dat'
            with open(filename, 'rb') as fid:
                D = np.fromfile(fid, dtype=np.float32)

            fs = D[10]
            self.dt = 1 / fs
            self.dx = D[13]
            self.nx = int(D[16])
            self.nt = int(fs * D[17])

            self.data = D[64:].reshape((self.nx, self.nt), order='F').T  # 使用Fortran顺序进行数据的reshape

            self.pre_data = self.data.copy()
            self.vmin = np.nanmin(self.data)
            self.vmax = np.nanmax(self.data)
            
            filepath = pathlib.Path(filename)
        else:
            raise ValueError("File Type Error!")

        self.radon_data = self.data.copy()
        self.pre_data = self.data.copy()
        
        # self.process()

        self.vmin = np.nanmin(self.data)
        self.vmax = np.nanmax(self.data)
    
    def imshowData(self):
        scale=0.1/(self.slider.value() + 1)
        vmin = self.vmin * scale
        vmax = self.vmax * scale

        self.fig.clear()
        self.ax = self.fig.add_subplot()
        self.ax.imshow(self.data, aspect="auto", cmap="rainbow", 
                       extent=[0, self.nx*self.dx, self.nt*self.dt, 0], 
                       vmin=vmin, vmax=vmax)
        self.ax.set_xlabel("Distance (m)")
        self.ax.set_ylabel("Time (s)")
        self.ax.set_title(self.files[self.ID].split('.')[0])
        self.canvas.draw()

    def next(self):
        self.points = []
        self.fig.clear()

        #self.fig.canvas.mpl_disconnect(lambda event: self.on_click(event, self.ax))

        self.ID += 1
        if self.ID >= len(self.files):
            self.ID = 0 
        self.fpath = os.path.join(self.folderName, self.files[self.ID])
        self.readData(self.fpath)
        self.imshowData()

    
    def moveData(self):
        pwd = os.getcwd()
        if not os.path.exists(f"{pwd}/done"):
            os.system(f"mkdir -p {pwd}/done")
        os.system(f"cp {str(self.fpath)} ./done/")



# def main():
#     app = QApplication(sys.argv)
#     ex = MainWindow()
    
#     sys.exit(app.exec())

# if __name__ == '__main__':
#     main()