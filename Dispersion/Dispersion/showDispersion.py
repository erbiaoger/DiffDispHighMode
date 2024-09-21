

import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QToolTip, QMessageBox,
                             QMainWindow, QHBoxLayout, QVBoxLayout, QFileDialog, QSizePolicy,
                             QSlider, QLabel, QLineEdit, QGridLayout, QGroupBox, QListWidget,
                             QTabWidget, QDialog, QCheckBox, QComboBox)
from PyQt6.QtGui import QIcon, QFont, QAction, QGuiApplication
from PyQt6.QtCore import Qt, QTimer, QFile, QTextStream, QSize
from PyQt6.QtWidgets import QApplication, QMainWindow, QDockWidget, QTextEdit



import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import dasQt.das as das
import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 16


from dasQt.utools.logPy3 import HandleLog
from dasQt.Dispersion.getDispersion.curves_class import pickPoints
from .caculateDispersion import CaculateDispersion


class DispersionMainWindow(QMainWindow):
    def __init__(self, MyProgram, sliderFig, title="Dispersion"):
        super().__init__()

        self.is_closed           : bool = False
        self.bool_saveCC         : bool = False
        self.bool_showDispersion : bool = False
        self.editFreqNorm        :str   = 'rma'
        self.editTimeNorm        :str   = 'no'
        self.editCCMethod        :str   = 'coherency'
        self.editSmethod         :str   = 'pws'
        
        self.logger = HandleLog(os.path.split(__file__)[-1].split(".")[0], path=os.getcwd(), level="DEBUG")
        self.MyProgram = CaculateDispersion(MyProgram)
        self.sliderFig = sliderFig

        self.setWindowTitle(title)
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

        self.initDispersion()

        mainFigure = QMainWindow()
        self.layout.addWidget(mainFigure, 1)

        widCCFigure = QWidget()
        dockCCFigure = QDockWidget("CC", self)
        dockCCFigure.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dockCCFigure.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dockCCFigure.setWidget(widCCFigure)
        mainFigure.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dockCCFigure)
        self.initCCFigure(widCCFigure)


        widDispersionFigure = QWidget()
        dockDispersionFigure = QDockWidget("Dispersion", self)
        dockDispersionFigure.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dockDispersionFigure.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dockDispersionFigure.setWidget(widDispersionFigure)
        mainFigure.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dockDispersionFigure)
        self.initDispersionFigure(widDispersionFigure)

        self.show()



    def initCCFigure(self, widFigure: QWidget) -> None:
        self.figCC = Figure()
        self.canvasCC = FigureCanvas(self.figCC)
        toolbar = NavigationToolbar(self.canvasCC, self)

        newLayout = QVBoxLayout()
        widFigure.setLayout(newLayout)
        newLayout.addWidget(self.canvasCC, 0)
        newLayout.addWidget(toolbar, 0)
        # self.layout.addWidget(widFigure, 1)


    def initDispersionFigure(self, widFigure: QWidget) -> None:
        self.figDispersion = Figure()
        self.canvasDispersion = FigureCanvas(self.figDispersion)
        toolbar = NavigationToolbar(self.canvasDispersion, self)

        newLayout = QVBoxLayout()
        widFigure.setLayout(newLayout)
        newLayout.addWidget(self.canvasDispersion, 0)
        newLayout.addWidget(toolbar, 0)
        # self.layout.addWidget(widFigure, 1)



    def initDispersion(self) -> None:
        widAll = QTabWidget()
        self.layout.addWidget(widAll, 0)
        layoutAll = QVBoxLayout()
        widAll.setLayout(layoutAll)

        widBtn = QGroupBox("CC", self)
        dispersionLayout = QVBoxLayout()
        widBtn.setLayout(dispersionLayout)
        # layoutAll.addWidget(widBtn, 0)
        widAll.addTab(widBtn, "CC")
        # TabWidget.tabBarClicked.connect(lambda: self.figTabWidget.setCurrentIndex(2))
        # TabWidget.addTab(widAll, "Dispersion")


        labFmin                = QLabel('fmin', self)
        labFmax                = QLabel('fmax', self)

        labFreqNorm = QLabel('freq norm', self)
        self.comboBoxFreqNorm = QComboBox()       # 创建下拉菜单
        self.comboBoxFreqNorm.addItem("rma")
        self.comboBoxFreqNorm.addItem("phase_only")
        self.comboBoxFreqNorm.addItem("no")
        self.comboBoxFreqNorm.activated.connect(self.on_comboBoxFreqNorm) # 连接信号

        labTimeNorm = QLabel('time norm', self)
        self.comboBoxTimeNorm = QComboBox()       # 创建下拉菜单
        self.comboBoxTimeNorm.addItem("no")
        self.comboBoxTimeNorm.addItem("rma")
        self.comboBoxTimeNorm.addItem("one_bit")
        self.comboBoxTimeNorm.activated.connect(self.on_comboBoxTimeNorm) # 连接信号

        labCCMethon = QLabel('cc method', self)
        self.comboBoxCCMethod = QComboBox()       # 创建下拉菜单
        self.comboBoxCCMethod.addItem("coherency")
        self.comboBoxCCMethod.addItem("deconv")
        self.comboBoxCCMethod.addItem("xcorr")
        self.comboBoxCCMethod.activated.connect(self.on_comboBoxCCMethod) # 连接信号

        labSmooth_N            = QLabel('smooth_N', self)
        labSmoothspect_N       = QLabel('smoothspect_N', self)
        labMaxlag              = QLabel('maxlag', self)
        labCC_len              = QLabel('CC_len', self)
        labCh1                 = QLabel('Xmin', self)
        labCh2                 = QLabel('Xmax', self)
        labShot                = QLabel('Shot', self)
        labSmethod = QLabel('Smethod', self)
        self.editFmin          = QLineEdit('5', self)
        self.editFmax          = QLineEdit('15', self)
        self.editSmooth_N      = QLineEdit('5', self)
        self.editSmoothspect_N = QLineEdit('5', self)
        self.editMaxlag        = QLineEdit('2.0', self)
        self.editCC_len        = QLineEdit('6', self)
        self.editCh1           = QLineEdit('180', self)
        self.editCh2           = QLineEdit('300', self)
        self.editShot          = QLineEdit('0', self)
        self.comboBoxSmethod = QComboBox()       # 创建下拉菜单
        self.comboBoxSmethod.addItem("pws")
        self.comboBoxSmethod.addItem("robust")
        self.comboBoxSmethod.addItem("linear")
        self.comboBoxSmethod.activated.connect(self.on_comboBoxSmethod) # 连接信号



        grid1 = QGridLayout()
        grid1.setSpacing(10)
        grid1.addWidget(labFmin, 1, 0)
        grid1.addWidget(self.editFmin, 1, 1)
        grid1.addWidget(labFmax, 2, 0)
        grid1.addWidget(self.editFmax, 2, 1)
        grid1.addWidget(labFreqNorm, 3, 0)
        grid1.addWidget(self.comboBoxFreqNorm, 3, 1)
        grid1.addWidget(labTimeNorm, 4, 0)
        grid1.addWidget(self.comboBoxTimeNorm, 4, 1)
        grid1.addWidget(labCCMethon, 5, 0)
        grid1.addWidget(self.comboBoxCCMethod, 5, 1)

        grid2 = QGridLayout()
        grid2.setSpacing(10)
        grid2.addWidget(labSmooth_N, 1, 0)
        grid2.addWidget(self.editSmooth_N, 1, 1)
        grid2.addWidget(labSmoothspect_N, 2, 0)
        grid2.addWidget(self.editSmoothspect_N, 2, 1)
        grid2.addWidget(labMaxlag, 3, 0)
        grid2.addWidget(self.editMaxlag, 3, 1)
        
        grid3 = QGridLayout()
        grid3.setSpacing(10)
        grid3.addWidget(labCC_len, 1, 0)
        grid3.addWidget(self.editCC_len, 1, 1)
        grid3.addWidget(labCh1, 2, 0)
        grid3.addWidget(self.editCh1, 2, 1)
        grid3.addWidget(labCh2, 3, 0)
        grid3.addWidget(self.editCh2, 3, 1)
        grid3.addWidget(labShot, 4, 0)
        grid3.addWidget(self.editShot, 4, 1)
        grid3.addWidget(labSmethod, 5, 0)
        grid3.addWidget(self.comboBoxSmethod, 5, 1)

        widFilter = QGroupBox("Filter", self)
        widFilter.setLayout(grid1)
        dispersionLayout.addWidget(widFilter, 1)

        widSmooth = QGroupBox("Smooth", self)
        widSmooth.setLayout(grid2)
        dispersionLayout.addWidget(widSmooth, 1)

        widCC = QGroupBox("CC", self)
        widCC.setLayout(grid3)
        dispersionLayout.addWidget(widCC, 1)

        btnCC = QPushButton('noise cc', self)
        btnCC.setToolTip('This is a <b>QPushButton</b> widget')
        btnCC.clicked.connect(
            lambda: [self.selfCaculateCC(), self.imshowCC()])
        
        btnCCAll = QPushButton('CC All', self)
        btnCCAll.setToolTip('This is a <b>QPushButton</b> widget')
        btnCCAll.clicked.connect(
            lambda: [self.imshowCC(bool_all=True)])

        btnNextData = QPushButton('Next Data', self)
        btnNextData.setToolTip('This is a <b>QPushButton</b> widget')
        btnNextData.clicked.connect(
            lambda: [self.readNextDataCC()])   # , self.imshowDataAll()

        btnNext20CC = QPushButton('Next more CC', self)
        btnNext20CC.setToolTip('This is a <b>QPushButton</b> widget')
        btnNext20CC.clicked.connect(
            lambda: [self.readNext20CC()])
        self.editNext20CC = QLineEdit('20', self)
        
        btnClearCC = QPushButton('Clear CC', self)
        btnClearCC.setToolTip('This is a <b>QPushButton</b> widget')
        btnClearCC.clicked.connect(
            lambda: [self.clearCC()])
        
        
        self.checkBox = QCheckBox('Save CC', self)
        self.checkBox.setChecked(False)
        self.checkBox.stateChanged.connect(self.checkBoxChanged)

        layGrid = QGridLayout()
        layGrid.setSpacing(5)
        layGrid.addWidget(btnCC, 1, 0)
        layGrid.addWidget(btnCCAll, 1, 1)
        layGrid.addWidget(btnNextData, 2, 0)
        layGrid.addWidget(self.checkBox, 2, 1)
        layGrid.addWidget(btnNext20CC, 3, 0)
        layGrid.addWidget(self.editNext20CC, 3, 1)
        layGrid.addWidget(btnClearCC, 4, 0)



        widGrid = QWidget()
        widGrid.setLayout(layGrid)
        dispersionLayout.addWidget(widGrid, 1)









        widDispersion = QGroupBox("Dispersion", self)
        dispersionLayout1 = QVBoxLayout()
        widDispersion.setLayout(dispersionLayout1)
        # layoutAll.addWidget(widDispersion, 0)
        widAll.addTab(widDispersion, "Dispersion")
        
        widGroup = QGroupBox("", self)
        dispersionLayout1.addWidget(widGroup, 0)
        labCmin = QLabel('Cmin', self)
        labCmax = QLabel('Cmax', self)
        labdc  = QLabel('dc', self)
        labfmin = QLabel('fmin', self)
        labfmax = QLabel('fmax', self)
        self.editCmin            = QLineEdit('10.0', self)
        self.editCmax            = QLineEdit('600.0', self)
        self.editdc              = QLineEdit('1.0', self)
        self.editfmin_dispersion = QLineEdit('3.0', self)
        self.editfmax_dispersion = QLineEdit('20.0', self)


        
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(labCmin, 1, 0)
        grid.addWidget(self.editCmin, 1, 1)
        grid.addWidget(labCmax, 2, 0)
        grid.addWidget(self.editCmax, 2, 1)
        grid.addWidget(labdc, 3, 0)
        grid.addWidget(self.editdc, 3, 1)
        grid.addWidget(labfmin, 4, 0)
        grid.addWidget(self.editfmin_dispersion, 4, 1)
        grid.addWidget(labfmax, 5, 0)
        grid.addWidget(self.editfmax_dispersion, 5, 1)
        widGroup.setLayout(grid)
        
        
        
        
        self.btnUp = QPushButton('Up CC', self)
        self.btnUp.setCheckable(True)
        self.btnUp.setToolTip('This is a <b>QPushButton</b> widget')
        self.btnUp.clicked.connect(lambda: [self.getUpCC()])
        
        self.btnDown = QPushButton('Down CC', self)
        self.btnDown.setCheckable(True)
        self.btnDown.setToolTip('This is a <b>QPushButton</b> widget')
        self.btnDown.clicked.connect(lambda: [self.getDownCC()])
        
        btnGetDispersion = QPushButton('Get Dispersion', self)
        btnGetDispersion.setToolTip('This is a <b>QPushButton</b> widget')
        btnGetDispersion.clicked.connect(lambda: [
            self.imshowDispersion(
                float(self.editCmin.text()), 
                float(self.editCmax.text()), 
                float(self.editdc.text()),
                float(self.editfmin_dispersion.text()), 
                float(self.editfmax_dispersion.text()))])

        btnGetAllDispersion = QPushButton('Get All Dispersion', self)
        btnGetAllDispersion.setToolTip('This is a <b>QPushButton</b> widget')
        btnGetAllDispersion.clicked.connect(lambda: [
            self.imshowDispersion(
                    float(self.editCmin.text()), 
                    float(self.editCmax.text()), 
                    float(self.editdc.text()),
                    float(self.editfmin_dispersion.text()), 
                    float(self.editfmax_dispersion.text()), 
                    bool_all=True)])

        btnNextData = QPushButton('Next Data', self)
        btnNextData.setToolTip('This is a <b>QPushButton</b> widget')
        btnNextData.clicked.connect(lambda: [
                self.readNextDispersion()])

        btnNext20Dispersion = QPushButton('Next 20 Data', self)
        btnNext20Dispersion.setToolTip('This is a <b>QPushButton</b> widget')
        btnNext20Dispersion.clicked.connect(
            lambda: [self.readNext20Dispersion()])
        # self.editNext20Dispersion = QLineEdit('20', self)
        
        self.checkBoxDispersion = QCheckBox('Save Dispersion', self)
        self.checkBoxDispersion.setChecked(False)
        self.checkBoxDispersion.stateChanged.connect(self.checkBoxChangedSaveDispersion)

        btnDispersionAll = QPushButton('Dispersion All', self)
        btnDispersionAll.setToolTip('This is a <b>QPushButton</b> widget')
        btnDispersionAll.clicked.connect(lambda: [
        self.imshowDispersion(
            float(self.editCmin.text()), 
            float(self.editCmax.text()), 
            float(self.editdc.text()),
            float(self.editfmin_dispersion.text()), 
            float(self.editfmax_dispersion.text()), 
            bool_all=True)])

        
        widGroup = QGroupBox("", self)
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(self.btnUp, 1, 0)
        grid.addWidget(self.btnDown, 1, 1)
        grid.addWidget(btnGetDispersion, 2, 0)
        grid.addWidget(btnGetAllDispersion, 2, 1)
        grid.addWidget(btnNextData, 3, 0)
        grid.addWidget(btnNext20Dispersion, 3, 1)
        # grid.addWidget(self.editNext20Dispersion,4, 1)
        grid.addWidget(self.checkBoxDispersion, 4, 0)
        grid.addWidget(btnDispersionAll, 4, 1)
        
        widGroup.setLayout(grid)
        dispersionLayout1.addWidget(widGroup, 0)
        


        labSkipNch = QLabel('skip_Nch', self)
        labSkipNt = QLabel('skip_Nt', self)
        labThreshold = QLabel('threshold', self)
        labMaxMode = QLabel('maxMode', self)
        labMinCarNum = QLabel('minCarNum', self)
        self.editSkipNch = QLineEdit('5', self)
        self.editSkipNt = QLineEdit('1', self)
        self.editThreshold = QLineEdit('0.5', self)
        self.editMaxMode = QLineEdit('10', self)
        self.editMinCarNum = QLineEdit('40', self)

        btnGetPoint = QPushButton('Auto Get Point', self)
        btnGetPoint.setToolTip('This is a <b>QPushButton</b> widget')
        btnGetPoint.clicked.connect(lambda: [
            self.imshowPoint()])
        
        btnPickPoints = QPushButton('Pick Points', self)
        btnPickPoints.setToolTip('This is a <b>QPushButton</b> widget')
        btnPickPoints.clicked.connect(lambda: [
            self.imshowPickPoints()])
        
        btnSaveClass = QPushButton('Save Class', self)
        btnSaveClass.setToolTip('This is a <b>QPushButton</b> widget')
        btnSaveClass.clicked.connect(lambda: [
            self.saveClass()])
        
        self.editSaveClass = QLineEdit('1', self)

        btnInterp = QPushButton('Interp', self)
        btnInterp.setToolTip('This is a <b>QPushButton</b> widget')
        btnInterp.clicked.connect(lambda: [
            self.MyProgram.interpPoints(), self.redraw()])
        
        saveCurve = QPushButton('Save Curve', self)
        saveCurve.setToolTip('This is a <b>QPushButton</b> widget')
        saveCurve.clicked.connect(lambda: [
            self.saveCurve()])
        
        widGroup = QGroupBox("", self)
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(labSkipNch, 1, 0)
        grid.addWidget(self.editSkipNch, 1, 1)
        grid.addWidget(labSkipNt, 2, 0)
        grid.addWidget(self.editSkipNt, 2, 1)
        grid.addWidget(labThreshold, 3, 0)
        grid.addWidget(self.editThreshold, 3, 1)
        grid.addWidget(labMaxMode, 4, 0)
        grid.addWidget(self.editMaxMode, 4, 1)
        grid.addWidget(labMinCarNum, 5, 0)
        grid.addWidget(self.editMinCarNum, 5, 1)
        grid.addWidget(btnGetPoint, 6, 0)
        grid.addWidget(btnPickPoints, 6, 1)
        grid.addWidget(btnSaveClass, 7, 0)
        grid.addWidget(self.editSaveClass, 7, 1)
        grid.addWidget(btnInterp, 8, 0)
        grid.addWidget(saveCurve, 8, 1)
        
        widGroup.setLayout(grid)
        dispersionLayout1.addWidget(widGroup, 0)




    def readNextDataCC(self):
        self.MyProgram.readNextData()
        self.selfCaculateCC()
        self.imshowCC()


    def readNextDispersion(self):
        self.MyProgram.readNextData()
        self.selfCaculateCC()
        self.imshowCC()
        
        self.imshowDispersion(
            float(self.editCmin.text()), 
            float(self.editCmax.text()), 
            float(self.editdc.text()),
            float(self.editfmin_dispersion.text()), 
            float(self.editfmax_dispersion.text()))

    def readNext20CC(self):
        for i in range(int(self.editNext20CC.text())):
            self.MyProgram.readNextData()
            self.selfCaculateCC()

            if i == int(self.editNext20CC.text()) - 1:
                self.MyProgram.caculateCCAll(str(self.editSmethod))
                self.imshowCC(bool_all=True)


    def readNext20Dispersion(self):
        for i in range(int(self.editNext20CC.text())):
            self.MyProgram.readNextData()
            self.selfCaculateCC()

            if i == int(self.editNext20CC.text()) - 1:
                self.MyProgram.caculateCCAll(str(self.editSmethod))
                self.imshowCC(bool_all=True)
                self.imshowDispersion(
                    float(self.editCmin.text()), 
                    float(self.editCmax.text()), 
                    float(self.editdc.text()),
                    float(self.editfmin_dispersion.text()), 
                    float(self.editfmax_dispersion.text()), 
                    bool_all=True)

    def clearCC(self):
        self.MyProgram.clearCC()

    def saveClass(self):
        class_num = int(self.editSaveClass.text())
        points = self.MyProgram.saveClass(class_num)
        self.MyProgram.points = points.tolist()


    def imshowDispersion(self, Cmin, Cmax, dc, fmin, fmax, bool_all=False):
        self.figDispersion.clear(); ax1 = self.figDispersion.add_subplot(111)
        ax1.cla()
        self.MyProgram.imshowDispersion(ax1, Cmin, Cmax, dc, fmin, fmax, bool_all)
        self.canvasDispersion.draw()

    
    def imshowPoint(self):
        self.figDispersion.clear(); ax1 = self.figDispersion.add_subplot(111)
        ax1.cla()

        self.MyProgram.imshowPoint(ax1, 
                        int(self.editSkipNch.text()),
                        int(self.editSkipNt.text()),
                        float(self.editThreshold.text()),
                        int(self.editMaxMode.text()),
                        int(self.editMinCarNum.text()))

        self.canvasDispersion.draw()



    def imshowPickPoints(self):
        self.saveClass()
        self.redraw()

        self.figDispersion.canvas.mpl_connect('button_press_event', self.onclickPoints)



    def selfCaculateCC(self):
        dispersion_parse = {
            'sps'           : 1/self.MyProgram.dt,    # current sampling rate
            'samp_freq'     : 1/self.MyProgram.dt,    # targeted sampling rate
            'iShot'         : int(self.editShot.text()),    # current shot index
            'freqmin'       : float(self.editFmin.text()),          # pre filtering frequency bandwidth预滤波频率带宽
            'freqmax'       : float(self.editFmax.text()),           # note this cannot exceed Nquist freq
            'freq_norm'     : str(self.editFreqNorm),        # 'no' for no whitening, or 'rma' for running-mean average, 'phase_only' for sign-bit normalization in freq domain.
            'time_norm'     : str(self.editTimeNorm),    # 'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain
            'cc_method'     : str(self.editCCMethod),      # 'xcorr' for pure cross correlation, 'deconv' for deconvolution;
            'smooth_N'      : int(self.editSmooth_N.text()),            # moving window length for time domain normalization if selected (points)
            'smoothspect_N' : int(self.editSmoothspect_N.text()),            # moving window length to smooth spectrum amplitude (points)
            'maxlag'        : float(self.editMaxlag.text()),          # lags of cross-correlation to save (sec)
            'max_over_std'  : 10**9,        # threahold to remove window of bad signals: set it to 10*9 if prefer not to remove them
            'cc_len'        : int(self.editCC_len.text()),            # correlate length in second(sec)
            'cha1'          : int(self.editCh1.text()),           # start channel index for the sub-array
            'cha2'          : int(self.editCh2.text()),           # end channel index for the sub-array
        }
        
        self.MyProgram.selfCaculateCC(dispersion_parse)


    def imshowCC(self, bool_all=False):
        self.figCC.clear(); ax1 = self.figCC.add_subplot(111)
        ax1.cla()

        ax1 = self.MyProgram.imshowCC(ax1, bool_all)
        self.canvasCC.draw()





    # TODO : CC data get up and down
    def getUpCC(self):
        if self.btnUp.isChecked():
            self.btnDown.setChecked(False)
            self.MyProgram.getDownOrUpCC(self.btnDown.isChecked())
            self.btnUp.setText('Stop')
            self.btnDown.setText('Down CC')
        else:
            self.btnUp.setText('Up CC')
    
    def getDownCC(self):
        if self.btnDown.isChecked():
            self.btnUp.setChecked(False)
            self.MyProgram.getDownOrUpCC(self.btnDown.isChecked())
            self.btnDown.setText('Stop')
            self.btnUp.setText('Up CC')
        else:
            self.btnDown.setText('Down CC')

    def saveCC(self, filename):
        self.MyProgram.saveCC(filename)

    def saveDispersion(self, filename):
        self.MyProgram.saveDispersion(filename)

    def saveCurve(self):
        filename = QFileDialog.getSaveFileName(self, 'Save File', '', 'NPZ files (*.npz)')
        self.MyProgram.saveCurve(filename[0])

    def on_comboBoxFreqNorm(self):
        self.editFreqNorm = self.comboBoxFreqNorm.currentText()

    def on_comboBoxTimeNorm(self):
        self.editTimeNorm = self.comboBoxTimeNorm.currentText()

    def on_comboBoxCCMethod(self):
        self.editCCMethod = self.comboBoxCCMethod.currentText()
    
    def on_comboBoxSmethod(self):
        self.editSmethod = self.comboBoxSmethod.currentText()



    def onclickPoints(self, event):
        threshold = 1000.
        if event.button == 1:  # 鼠标左键
            self.MyProgram.points.append([event.xdata, event.ydata])
            self.redraw()
        elif event.button == 3:  # 鼠标右键
            if not self.MyProgram.points:
                return
            x, y = zip(*self.MyProgram.points)
            distances = [(event.xdata - xi) ** 2 + (event.ydata - yi) ** 2 for xi, yi in zip(x, y)]
            min_distance = min(distances)
            if min_distance < threshold:
                index = distances.index(min_distance)
                del self.MyProgram.points[index]
                self.redraw()

    def redraw(self):
        self.figDispersion.clear(); ax1 = self.figDispersion.add_subplot(111)
        ax1.cla()
        self.MyProgram.imshowPointNone(ax1, self.MyProgram.points)
        print('redraw')
        self.canvasDispersion.draw()


    def closeEvent(self, event):
        """重写关闭事件"""
        self.is_closed = True
        print("窗口已关闭")
        event.accept()  # 接受关闭事件，完成窗口关闭
        
    def isVisible(self):
        return self.is_closed

    def checkBoxChanged(self, state):
        if state == 0:
            self.bool_saveCC = False
            self.MyProgram.bool_saveCC = False
        else:
            self.bool_saveCC = True
            self.MyProgram.bool_saveCC = True
    
    def checkBox_fig_Changed(self, state):
        if state == 0:
            self.bool_saveFig = False
            self.MyProgram.bool_saveFig = False
        else:
            self.bool_saveFig = True
            self.MyProgram.bool_saveFig = True

    def checkBoxChangedSaveDispersion(self, state):
        if state == 0:
            self.bool_saveDispersion = False
            self.MyProgram.bool_saveDispersion = False
        else:
            self.bool_saveDispersion = True
            self.MyProgram.bool_saveDispersion = True









if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DispersionMainWindow(das.DAS())
    window.show()
    sys.exit(app.exec())