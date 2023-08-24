import sys
import threading

import cv2
import serial
from PyQt5.QtCore import QByteArray, Qt
from PyQt5.QtGui import QFont, QImage, QMovie, QPixmap
from PyQt5.QtWidgets import (
     QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)

from PyQt5.QtWidgets import *
from PyQt5 import uic
from buri_funcions import attachBuri
form_class = uic.loadUiType("app.ui")[0]

#화면을 띄우는데 사용되는 Class 선언
class WindowClass(QWidget, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('두루미 코스프레')
        self.initUI()
        self.start_cam()
        self.pic_num = 0
        
    def initUI(self) :
        self.btn_shot.clicked.connect(self.shot)
        self.btn_save.clicked.connect(self.save)
        
    def opencv(self):
        self.running = True
        cap = cv2.VideoCapture(0)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(width, height)
        self.cam_label.resize(int(width), int(height))

        while self.running:
            ret, self.img = cap.read()
            if ret:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                self.img = cv2.flip(self.img, 1)
                # print(self.img.shape)
                
                h, w, c = self.img.shape
                qImg = QImage(
                    self.img.data,
                    w,
                    h,
                    w*c,
                    QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qImg)
                self.cam_label.setPixmap(pixmap)

        cap.release()
        print("Thread end.")

    def start_cam(self):
        self.running = True
        th = threading.Thread(target=self.opencv)
        th.start()
        print("started..")
        
    def shot(self) :
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        name = f'picture/{self.pic_num}.jpg'
        cv2.imwrite(name, self.img)
        self.pic_num += 1
        self.img_buri = attachBuri(f'picture/{self.pic_num-1}.jpg')
        self.res_label.setPixmap(QPixmap(self.img_buri))
        
    def save(self) :
        cv2.imwrite(f'picture/{self.pic_num-1}_buri.jpg',self.img_buri)
        
if __name__ == "__main__" :
    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv) 

    #WindowClass의 인스턴스 생성
    myWindow = WindowClass() 

    #프로그램 화면을 보여주는 코드
    myWindow.show()

    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()
