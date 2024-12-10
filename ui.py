import sys
import cv2
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLCDNumber, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QTimer
from StartStreamCamera_Running import *

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt5 Camera App with LCD Display")
        self.setGeometry(100, 100, 800, 600)

        self.follow_timer = 0

        # Layouts
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        # LCD Display Widgets
        self.lcd1 = QLCDNumber()
        self.lcd2 = QLCDNumber()
        self.lcd3 = QLCDNumber()
        self.lcd4 = QLCDNumber()

        self.lcd1.display(123)
        self.lcd2.display(456)
        self.lcd3.display(789)
        self.lcd4.display(0)

        # Camera display
        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)

        # Buttons
        self.button1 = QPushButton("Start Camera")
        self.button2 = QPushButton("Stop Camera")
        self.button3 = QPushButton("Reset LCD")
        self.button4 = QPushButton("Exit")

        self.button1.clicked.connect(self.start_camera)
        self.button2.clicked.connect(self.stop_camera)
        self.button3.clicked.connect(self.reset_lcd)
        self.button4.clicked.connect(self.close)

        # Add buttons to the button layout
        button_layout.addWidget(self.button1)
        button_layout.addWidget(self.button2)
        button_layout.addWidget(self.button3)
        button_layout.addWidget(self.button4)

        # Add LCD displays to the main layout
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.lcd1)
        main_layout.addWidget(self.lcd2)
        main_layout.addWidget(self.lcd3)
        main_layout.addWidget(self.lcd4)

        self.setLayout(main_layout)

        # Camera setup
        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.exportimage)

    def start_camera(self):
        """Start the camera feed."""
        self.capture = export_image()    
        self.timer.start(20)

    def stop_camera(self):
        """Stop the camera feed."""
        self.timer.stop()
        self.camera_label.clear()

    def reset_lcd(self):
        """Reset all LCDs to default values."""
        self.lcd1.display(0)
        self.lcd2.display(0)
        self.lcd3.display(0)
        self.lcd4.display(0)

    def exportimage(self):
            self.follow_timer += 1
            if self.follow_timer >= 17:
                self.follow_timer =0
            
            # Tạo một ảnh màu xanh lam
            frame0 = export_image()
            frame = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)

            frame_rgb_main = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # Convert the frame to QImage
            image_main = QImage(frame_rgb_main.data, frame_rgb_main.shape[1], frame_rgb_main.shape[0],
                            frame_rgb_main.strides[0], QImage.Format_RGB888)
            
            self.camera_label.setPixmap(QPixmap.fromImage(image_main))

if __name__ == "__main__":
    startcamera()
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
