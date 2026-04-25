import os
import sys
import cv2
import math
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QProgressBar, QSlider)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from src.detector import BulletDetector

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    size_signal = pyqtSignal(int, int) # Signal baru untuk mengirim resolusi video
    count_signal = pyqtSignal(int)
    score_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.source = 0
        self.detector = BulletDetector('weights/bullet_model.pt')
        self.matrix = None
        self.ellipse_params = None 
        self.apply_ellipse_flag = False
        self.is_calibrated = False

    def apply_calibration_ellipse(self):
        self.apply_ellipse_flag = True

    def set_source(self, source):
        self.source = source

    def calculate_score_circle(self, hole_center, target_center, radius):
        if radius <= 0: return 0
        dx = hole_center[0] - target_center[0]
        dy = hole_center[1] - target_center[1]
        distance = math.sqrt(dx**2 + dy**2)
        normalized_dist = distance / radius
        
        if normalized_dist <= 0.5: return 10
        elif normalized_dist <= 1.0: return 9
        else:
            score = 8 - int((normalized_dist - 1.0) / 0.5)
            return max(0, score)

    def run(self):
        cap = cv2.VideoCapture(self.source)
        self.original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Kirim resolusi ke GUI agar slider menyesuaikan
        self.size_signal.emit(self.original_width, self.original_height)
        
        while self._run_flag:
            ret, frame = cap.read()
            if not ret or not self._run_flag: break

            try:
                if self.matrix is not None:
                    working_frame = cv2.warpPerspective(frame, self.matrix, (500, 500))
                else:
                    working_frame = frame

                annotated_frame, boxes_data = self.detector.detect_frame(working_frame)
                
                hole_centers = []
                target_center = None
                target_radius = 0 

                for cls, x, y, w, h in boxes_data:
                    if cls == 1: 
                        hole_centers.append((x, y))
                    elif cls == 0: 
                        target_center = (x, y) 
                        avg_size = (w + h) / 2
                        target_radius = (avg_size / 2) * 1.35

                current_total_score = 0
                if target_center is not None:
                    if self.is_calibrated and target_radius > 0:
                        for i in range(1, 11):
                            current_radius = int(target_radius * (i/2))
                            cv2.circle(annotated_frame, (int(target_center[0]), int(target_center[1])), current_radius, (0, 255, 0), 1)
                        for hole in hole_centers:
                            current_total_score += self.calculate_score_circle(hole, target_center, target_radius)

                    if self.ellipse_params is not None:
                        cx, cy = int(target_center[0]), int(target_center[1])
                        d_top, d_right, d_bottom, d_left = self.ellipse_params
                        
                        pt_top = (cx, cy - d_top)
                        pt_right = (cx + d_right, cy)
                        pt_bottom = (cx, cy + d_bottom)
                        pt_left = (cx - d_left, cy)
                        
                        ellipse_cx = cx + int((d_right - d_left) / 2)
                        ellipse_cy = cy + int((d_bottom - d_top) / 2)
                        ellipse_rx = int((d_left + d_right) / 2)
                        ellipse_ry = int((d_top + d_bottom) / 2)
                        cv2.ellipse(annotated_frame, (ellipse_cx, ellipse_cy), (ellipse_rx, ellipse_ry), 0, 0, 360, (255, 255, 0), 2)
                        cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

                        if self.apply_ellipse_flag:
                            pts1 = np.float32([pt_top, pt_right, pt_bottom, pt_left])
                            pts2 = np.float32([[250, 50], [450, 250], [250, 450], [50, 250]])
                            self.matrix = cv2.getPerspectiveTransform(pts1, pts2)
                            self.apply_ellipse_flag = False
                            self.ellipse_params = None
                            self.is_calibrated = True

                self.count_signal.emit(len(hole_centers))
                self.score_signal.emit(current_total_score)
                
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h_f, w_f, ch = rgb_image.shape
                qt_img = QImage(rgb_image.data, w_f, h_f, ch*w_f, QImage.Format.Format_RGB888).copy()
                self.change_pixmap_signal.emit(qt_img)

            except Exception as e:
                print(f"Error: {e}")
                continue
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bullet Hole Detection System")
        self.setMinimumSize(1000, 850)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.image_label.setFixedSize(850, 480)
        self.main_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.info_layout = QHBoxLayout()
        self.info_label = QLabel("Jumlah Lubang: 0", self)
        self.score_label = QLabel("Total Skor: 0", self)
        label_style = "font-size: 18px; font-weight: bold; color: #ffffff; padding: 10px;"
        self.info_label.setStyleSheet(label_style)
        self.score_label.setStyleSheet(label_style)
        self.info_layout.addWidget(self.info_label)
        self.info_layout.addWidget(self.score_label)
        self.main_layout.addLayout(self.info_layout)

        # PANEL KALIBRASI
        self.panel_calib = QWidget()
        self.layout_calib = QVBoxLayout(self.panel_calib)
        
        grid_sliders = QHBoxLayout()
        v_col = QVBoxLayout()
        self.top_label = QLabel("Jarak Atas: 0 px")
        self.slider_top = QSlider(Qt.Orientation.Horizontal)
        self.bottom_label = QLabel("Jarak Bawah: 0 px")
        self.slider_bottom = QSlider(Qt.Orientation.Horizontal)
        v_col.addWidget(self.top_label)
        v_col.addWidget(self.slider_top)
        v_col.addWidget(self.bottom_label)
        v_col.addWidget(self.slider_bottom)

        h_col = QVBoxLayout()
        self.right_label = QLabel("Jarak Kanan: 0 px")
        self.slider_right = QSlider(Qt.Orientation.Horizontal)
        self.left_label = QLabel("Jarak Kiri: 0 px")
        self.slider_left = QSlider(Qt.Orientation.Horizontal)
        h_col.addWidget(self.right_label)
        h_col.addWidget(self.slider_right)
        h_col.addWidget(self.left_label)
        h_col.addWidget(self.slider_left)

        grid_sliders.addLayout(v_col)
        grid_sliders.addLayout(h_col)
        self.layout_calib.addLayout(grid_sliders)

        self.slider_top.valueChanged.connect(self.update_ellipse_params)
        self.slider_bottom.valueChanged.connect(self.update_ellipse_params)
        self.slider_right.valueChanged.connect(self.update_ellipse_params)
        self.slider_left.valueChanged.connect(self.update_ellipse_params)

        self.btn_apply = QPushButton("✅ Terapkan Kalibrasi")
        self.btn_apply.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; height: 35px;")
        self.btn_apply.clicked.connect(self.apply_transform)
        self.layout_calib.addWidget(self.btn_apply)

        self.panel_calib.hide()
        self.main_layout.addWidget(self.panel_calib)

        self.btn_layout = QHBoxLayout()
        self.btn_webcam = QPushButton("Mulai Webcam")
        self.btn_webcam.clicked.connect(self.start_webcam)
        self.btn_video = QPushButton("Buka File Video")
        self.btn_video.clicked.connect(self.start_video)
        self.btn_stop = QPushButton("Stop / Reset")
        self.btn_stop.clicked.connect(self.stop_process)
        
        self.btn_layout.addWidget(self.btn_webcam)
        self.btn_layout.addWidget(self.btn_video)
        self.btn_layout.addWidget(self.btn_stop)
        self.main_layout.addLayout(self.btn_layout)

        self.thread = None

    def start_webcam(self):
        self.stop_process()
        self.thread = VideoThread()
        self.thread.set_source(0)
        self.connect_thread()
        self.thread.start()

    def start_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Video", "", "Video Files (*.mp4 *.avi *.mov *.jpeg)")
        if file_path:
            self.stop_process()
            self.thread = VideoThread()
            self.thread.set_source(os.path.normpath(file_path))
            self.connect_thread()
            self.thread.start()

    def connect_thread(self):
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.size_signal.connect(self.setup_slider_ranges) # Hubungkan signal resolusi
        self.thread.count_signal.connect(self.update_count)
        self.thread.score_signal.connect(self.update_score)

    def setup_slider_ranges(self, width, height):
        """Mengatur batas slider sesuai resolusi video yang masuk"""
        self.panel_calib.show()
        # Batas maksimal slider sekarang dinamis mengikuti ukuran video
        self.slider_top.setRange(5, height)
        self.slider_bottom.setRange(5, height)
        self.slider_right.setRange(5, width)
        self.slider_left.setRange(5, width)
        
        # Nilai awal default (biar tidak 0)
        self.slider_top.setValue(height // 4)
        self.slider_bottom.setValue(height // 4)
        self.slider_right.setValue(width // 4)
        self.slider_left.setValue(width // 4)

    def update_image(self, qt_img):
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(scaled)

    def update_count(self, count):
        self.info_label.setText(f"Jumlah Lubang: {count}")
    
    def update_score(self, score):
        self.score_label.setText(f"Total Skor: {score}")

    def update_ellipse_params(self):
        t, b, r, l = self.slider_top.value(), self.slider_bottom.value(), self.slider_right.value(), self.slider_left.value()
        self.top_label.setText(f"Jarak Atas: {t} px")
        self.bottom_label.setText(f"Jarak Bawah: {b} px")
        self.right_label.setText(f"Jarak Kanan: {r} px")
        self.left_label.setText(f"Jarak Kiri: {l} px")
        if self.thread:
            self.thread.ellipse_params = (t, r, b, l)

    def apply_transform(self):
        if self.thread:
            self.thread.apply_calibration_ellipse()
            self.panel_calib.hide()

    def stop_process(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.image_label.clear()
        self.panel_calib.hide()
        self.info_label.setText("Jumlah Lubang: 0")
        self.score_label.setText("Total Skor: 0")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec())