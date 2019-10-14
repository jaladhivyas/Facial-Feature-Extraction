from imutils import face_utils
import argparse
import imutils
import dlib
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
global dead

class CloneThread(QtCore.QThread):
    signal = QtCore.pyqtSignal('PyQt_PyObject')

    def __init__(self, myDisplay, videoCapture):
        QtCore.QThread.__init__(self)
        self.myDisplay    = myDisplay
        self.videoCapture = videoCapture

    # run method gets called when we start the thread
    def run(self):
        global dead
        dead = False
        ap = argparse.ArgumentParser()
        ap.add_argument("-p", "--shape-predictor", required=True,
                        help="path to facial landmark predictor")

        args = vars(ap.parse_args())

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        detector  = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(args["shape_predictor"])

        while (not dead):
            # grab the frame from the threaded video stream, resize it to
            # have a maximum width of 400 pixels, and convert it to
            # grayscale
            ret, frame = self.videoCapture.read()
            frame = imutils.resize(frame, width=400)
            if (frame.shape[2]) == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = detector(gray, 0)
            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                frame = QtGui.QImage(frame, frame.shape[1], frame.shape[0],qformat)
                frameout = frame.rgbSwapped()
                self.myDisplay.setPixmap(QtGui.QPixmap.fromImage(frameout))
                self.myDisplay.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Feature extractor")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.imageDisplay = QtWidgets.QLabel(self.centralwidget)
        self.imageDisplay.setGeometry(QtCore.QRect(160, 60, 471, 321))
        self.imageDisplay.setAutoFillBackground(False)
        self.imageDisplay.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imageDisplay.setLineWidth(4)
        self.imageDisplay.setMidLineWidth(2)
        self.imageDisplay.setText("")
        self.imageDisplay.setObjectName("imageDisplay")
        self.Start = QtWidgets.QPushButton(self.centralwidget)
        self.Start.setGeometry(QtCore.QRect(240, 420, 99, 27))
        self.Start.setObjectName("Start")
        self.Stop = QtWidgets.QPushButton(self.centralwidget)
        self.Stop.setGeometry(QtCore.QRect(470, 420, 99, 27))
        self.Stop.setObjectName("Stop")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Start.setText(_translate("MainWindow", "Start"))
        self.Stop.setText(_translate("MainWindow", "Stop"))
        self.Start.clicked.connect(self.onStartClicked)
        self.Stop.clicked.connect(self.onStopClicked)
        # Connect the signal from the thread to the finished method
        #self.videoThread.signal.connect(self.onStopClicked)

    def onStartClicked(self):

        # loop over the frames from the video stream
        qformat = QtGui.QImage.Format_Indexed8
        global capture
        capture =  cv2.VideoCapture(0)
        self.videoThread = CloneThread(self.imageDisplay, capture)
        self.Start.setEnabled(False)
        global dead
        dead = False
        self.videoThread.start()

    def onStopClicked(self):
        MainWindow.close()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

