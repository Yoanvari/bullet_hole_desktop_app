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
    progress_signal = pyqtSignal(int)
    count_signal = pyqtSignal(int)
    score_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.source = 0
        self.detector = BulletDetector('weights/bullet_model.pt')
        self.calibration_points = None
        self.matrix = None
        self.is_angle_mode = False
        
        self.show_crosshair = False
        
        self.ellipse_params = None 
        self.apply_ellipse_flag = False
        
        self.is_calibrated = False
    
    # 1. Metode Kalibrasi 4 Sudut
    def set_calibration_corners(self, points):
        self.is_angle_mode = False
        self.show_crosshair = False
        self.ellipse_params = None
        pts1 = np.float32(points)
        padding = 50
        pts2 = np.float32([
            [padding, padding], 
            [500 - padding, padding], 
            [500 - padding, 500 - padding], 
            [padding, 500 - padding]
        ])
        self.matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.is_calibrated = True

    # 2. Metode Kalibrasi 4 Sisi Lingkaran
    def set_calibration_circle(self, points):
        self.is_angle_mode = False
        self.show_crosshair = False
        self.ellipse_params = None
        pts1 = np.float32(points)
        pts2 = np.float32([
            [250, 50],   # Atas
            [450, 250],  # Kanan
            [250, 450],  # Bawah
            [50, 250]    # Kiri
        ])
        self.matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.is_calibrated = True

    # 3. Metode Kalibrasi Derajat
    def set_calibration_by_angle(self, pitch_degrees, yaw_degrees):
        self.show_crosshair = False
        self.ellipse_params = None
        w = self.original_width
        h = self.original_height
        
        f = max(w, h) 
        pitch_rad = math.radians(pitch_degrees)
        yaw_rad = math.radians(yaw_degrees)
        
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float32)
        
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(-pitch_rad), -math.sin(-pitch_rad)],
            [0, math.sin(-pitch_rad), math.cos(-pitch_rad)]
        ], dtype=np.float32)

        R_y = np.array([
            [math.cos(-yaw_rad), 0, math.sin(-yaw_rad)],
            [0, 1, 0],
            [-math.sin(-yaw_rad), 0, math.cos(-yaw_rad)]
        ], dtype=np.float32)

        R_combined = R_x @ R_y
        K_inv = np.linalg.inv(K)
        H_raw = K @ R_combined @ K_inv
        
        center_point = np.array([w/2, h/2, 1.0])
        new_center = H_raw @ center_point
        new_center = new_center / new_center[2]
        
        shift_x = w/2 - new_center[0]
        shift_y = h/2 - new_center[1]
        
        Translation_Matrix = np.array([[1, 0, shift_x], [0, 1, shift_y], [0, 0, 1]], dtype=np.float32)
        H_centered = Translation_Matrix @ H_raw
        
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T 
        warped_corners = H_centered @ corners
        warped_corners = warped_corners / warped_corners[2, :] 
        
        min_x = np.min(warped_corners[0, :])
        max_x = np.max(warped_corners[0, :])
        min_y = np.min(warped_corners[1, :])
        max_y = np.max(warped_corners[1, :])
        
        warped_w = max_x - min_x
        warped_h = max_y - min_y
        
        scale_x = w / warped_w if warped_w > w else 1.0
        scale_y = h / warped_h if warped_h > h else 1.0
        scale = min(scale_x, scale_y)
        
        Scale_Matrix = np.array([[scale, 0, (w/2) * (1 - scale)], [0, scale, (h/2) * (1 - scale)], [0, 0, 1]], dtype=np.float32)
        
        self.matrix = Scale_Matrix @ H_centered
        self.is_angle_mode = True
        self.is_calibrated = True

    # 4. Trigger untuk Metode 4
    def apply_calibration_ellipse(self):
        self.apply_ellipse_flag = True
        self.is_angle_mode = False
        self.show_crosshair = False

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
        
        while self._run_flag:
            ret, frame = cap.read()
            if not ret or not self._run_flag: break

            try:
                # A. Warping
                if self.matrix is not None:
                    if self.is_angle_mode:
                        working_frame = cv2.warpPerspective(frame, self.matrix, (self.original_width, self.original_height))
                    else:
                        working_frame = cv2.warpPerspective(frame, self.matrix, (500, 500))
                else:
                    working_frame = frame

                # B. Deteksi
                annotated_frame, boxes_data = self.detector.detect_frame(working_frame)
                
                # C. Olah Data
                hole_centers = []
                target_center = None
                target_radius = 0 

                for cls, x, y, w, h in boxes_data:
                    if cls == 1: 
                        hole_centers.append((x, y))
                    elif cls == 0: 
                        target_center = (x, y) # PUSAT DARI YOLO
                        avg_size = (w + h) / 2
                        target_radius = (avg_size / 2) * 1.35

                # D. Hitung Skor & Visualisasi
                current_total_score = 0
                if target_center is not None:
                    
                    # TAMPILKAN CINCIN SKOR HANYA JIKA SUDAH DIKALIBRASI
                    if self.is_calibrated and target_radius > 0:
                        for i in range(1, 11):
                            current_radius = int(target_radius * (i/2))
                            cv2.circle(annotated_frame, (int(target_center[0]), int(target_center[1])), current_radius, (0, 255, 0), 1)

                        for hole in hole_centers:
                            current_total_score += self.calculate_score_circle(hole, target_center, target_radius)

                    # Garis pemandu klik (Metode 2) tetap bisa tampil sebelum kalibrasi
                    if self.show_crosshair:
                        cx, cy = int(target_center[0]), int(target_center[1])
                        h_f, w_f = annotated_frame.shape[:2]
                        cv2.line(annotated_frame, (cx, 0), (cx, h_f), (0, 255, 255), 2)
                        cv2.line(annotated_frame, (0, cy), (w_f, cy), (0, 255, 255), 2)

                    # --- LOGIKA M4: TRACKING TITIK TENGAH ---
                    if self.ellipse_params is not None:
                        # Ambil posisi pusat black_contour secara dinamis
                        cx, cy = int(target_center[0]), int(target_center[1])
                        d_top, d_right, d_bottom, d_left = self.ellipse_params
                        
                        # Hitung 4 koordinat sisi terluar
                        pt_top = (cx, cy - d_top)
                        pt_right = (cx + d_right, cy)
                        pt_bottom = (cx, cy + d_bottom)
                        pt_left = (cx - d_left, cy)
                        
                        # 1. Hitung titik pusat elips bayangan
                        ellipse_cx = cx + int((d_right - d_left) / 2)
                        ellipse_cy = cy + int((d_bottom - d_top) / 2)
                        
                        # 2. Hitung radius rata-rata
                        ellipse_rx = int((d_left + d_right) / 2)
                        ellipse_ry = int((d_top + d_bottom) / 2)
                        
                        # 3. Gambar elips warna Cyan yang membungkus 4 titik
                        cv2.ellipse(annotated_frame, (ellipse_cx, ellipse_cy), (ellipse_rx, ellipse_ry), 0, 0, 360, (255, 255, 0), 2)
                        
                        # Menggambar garis tipis pemandu
                        cv2.line(annotated_frame, (cx, cy), pt_top, (255, 0, 255), 1)
                        cv2.line(annotated_frame, (cx, cy), pt_right, (255, 0, 255), 1)
                        cv2.line(annotated_frame, (cx, cy), pt_bottom, (255, 0, 255), 1)
                        cv2.line(annotated_frame, (cx, cy), pt_left, (255, 0, 255), 1)
                        
                        cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

                        # Jika tombol "Terapkan" ditekan
                        if self.apply_ellipse_flag:
                            pts1 = np.float32([pt_top, pt_right, pt_bottom, pt_left])
                            # Paksa 4 sisi menjadi lingkaran dengan padding 50 piksel
                            pts2 = np.float32([[250, 50], [450, 250], [250, 450], [50, 250]])
                            self.matrix = cv2.getPerspectiveTransform(pts1, pts2)
                            
                            self.apply_ellipse_flag = False
                            self.ellipse_params = None
                            self.is_calibrated = True # Sinyal mulai menggambar cincin skor

                # E. Emit Signal
                self.count_signal.emit(len(hole_centers))
                self.score_signal.emit(current_total_score)
                
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_img = QImage(rgb_image.data, w, h, ch*w, QImage.Format.Format_RGB888).copy()
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
        self.setMinimumSize(1000, 800)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.image_label.setFixedSize(850, 480)
        self.main_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.pbar = QProgressBar(self)
        self.pbar.setValue(0)
        self.main_layout.addWidget(self.pbar)

        self.info_layout = QHBoxLayout()
        self.info_label = QLabel("Jumlah Lubang: 0", self)
        self.score_label = QLabel("Total Skor: 0", self)
        label_style = "font-size: 18px; font-weight: bold; color: #ffffff; padding: 10px;"
        self.info_label.setStyleSheet(label_style)
        self.score_label.setStyleSheet(label_style)
        self.info_layout.addWidget(self.info_label)
        self.info_layout.addWidget(self.score_label)
        self.main_layout.addLayout(self.info_layout)

        # ==========================================
        # CONTAINER PANEL M3 (SLIDER DERAJAT)
        # ==========================================
        self.panel_m3 = QWidget()
        self.layout_m3 = QVBoxLayout(self.panel_m3)
        self.layout_m3.setContentsMargins(0, 0, 0, 0)
        
        self.pitch_label = QLabel("Kemiringan Vertikal (Pitch): 0°", self)
        self.pitch_label.setStyleSheet("color: white; font-weight: bold;")
        self.slider_pitch = QSlider(Qt.Orientation.Horizontal)
        self.slider_pitch.setRange(-90, 90)
        self.slider_pitch.setValue(0)
        self.slider_pitch.valueChanged.connect(self.update_angle_from_sliders)
        
        self.yaw_label = QLabel("Kemiringan Horizontal (Yaw): 0°", self)
        self.yaw_label.setStyleSheet("color: white; font-weight: bold;")
        self.slider_yaw = QSlider(Qt.Orientation.Horizontal)
        self.slider_yaw.setRange(-90, 90)
        self.slider_yaw.setValue(0)
        self.slider_yaw.valueChanged.connect(self.update_angle_from_sliders)
        
        self.layout_m3.addWidget(self.pitch_label)
        self.layout_m3.addWidget(self.slider_pitch)
        self.layout_m3.addWidget(self.yaw_label)
        self.layout_m3.addWidget(self.slider_yaw)
        
        self.panel_m3.hide() 
        self.main_layout.addWidget(self.panel_m3)

        # ==========================================
        # CONTAINER PANEL M4 (SLIDER TRACKING 4 SISI)
        # ==========================================
        self.panel_m4 = QWidget()
        self.layout_m4 = QVBoxLayout(self.panel_m4)
        self.layout_m4.setContentsMargins(0, 0, 0, 0)
        
        self.top_label = QLabel("Jarak Atas: 0 px", self)
        self.top_label.setStyleSheet("color: cyan; font-weight: bold;")
        self.slider_top = QSlider(Qt.Orientation.Horizontal)
        self.slider_top.valueChanged.connect(self.update_ellipse_from_sliders)
        
        self.bottom_label = QLabel("Jarak Bawah: 0 px", self)
        self.bottom_label.setStyleSheet("color: cyan; font-weight: bold;")
        self.slider_bottom = QSlider(Qt.Orientation.Horizontal)
        self.slider_bottom.valueChanged.connect(self.update_ellipse_from_sliders)

        self.right_label = QLabel("Jarak Kanan: 0 px", self)
        self.right_label.setStyleSheet("color: yellow; font-weight: bold;")
        self.slider_right = QSlider(Qt.Orientation.Horizontal)
        self.slider_right.valueChanged.connect(self.update_ellipse_from_sliders)

        self.left_label = QLabel("Jarak Kiri: 0 px", self)
        self.left_label.setStyleSheet("color: yellow; font-weight: bold;")
        self.slider_left = QSlider(Qt.Orientation.Horizontal)
        self.slider_left.valueChanged.connect(self.update_ellipse_from_sliders)

        self.btn_apply_m4 = QPushButton("✅ Terapkan Transformasi Target")
        self.btn_apply_m4.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 5px;")
        self.btn_apply_m4.clicked.connect(self.apply_ellipse_transform)

        grid_layout = QHBoxLayout()
        col1 = QVBoxLayout()
        col1.addWidget(self.top_label)
        col1.addWidget(self.slider_top)
        col1.addWidget(self.bottom_label)
        col1.addWidget(self.slider_bottom)
        
        col2 = QVBoxLayout()
        col2.addWidget(self.right_label)
        col2.addWidget(self.slider_right)
        col2.addWidget(self.left_label)
        col2.addWidget(self.slider_left)
        
        grid_layout.addLayout(col1)
        grid_layout.addLayout(col2)

        self.layout_m4.addLayout(grid_layout)
        self.layout_m4.addWidget(self.btn_apply_m4)
        
        self.panel_m4.hide() 
        self.main_layout.addWidget(self.panel_m4)

        # ==========================================
        # TOMBOL KONTROL UTAMA
        # ==========================================
        self.btn_layout = QHBoxLayout()
        self.btn_webcam = QPushButton("Webcam")
        self.btn_webcam.clicked.connect(self.start_webcam)
        self.btn_video = QPushButton("File Video")
        self.btn_video.clicked.connect(self.start_video)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_process)
        self.btn_layout.addWidget(self.btn_webcam)
        self.btn_layout.addWidget(self.btn_video)
        self.btn_layout.addWidget(self.btn_stop)
        self.main_layout.addLayout(self.btn_layout)

        # --- TOMBOL-TOMBOL METODE KALIBRASI ---
        self.calib_layout = QHBoxLayout()
        self.thread = None
        self.calibration_points = []
        self.is_calibration_mode = False
        
        self.btn_calib_corners = QPushButton("1. Klik 4 Sudut")
        self.btn_calib_corners.clicked.connect(self.start_calib_corners)
        
        self.btn_calib_circle = QPushButton("2. Klik Sisi Lingkaran")
        self.btn_calib_circle.clicked.connect(self.start_calib_circle)

        self.btn_angle = QPushButton("3. Slider Derajat 3D")
        self.btn_angle.clicked.connect(self.start_calib_angle)

        self.btn_ellipse = QPushButton("4. Slider Target (Auto Center)")
        self.btn_ellipse.clicked.connect(self.start_calib_ellipse)
        
        self.calib_layout.addWidget(self.btn_calib_corners)
        self.calib_layout.addWidget(self.btn_calib_circle)
        self.calib_layout.addWidget(self.btn_angle)
        self.calib_layout.addWidget(self.btn_ellipse)
        self.main_layout.addLayout(self.calib_layout)

        self.image_label.mousePressEvent = self.get_pixel_position

    def start_webcam(self):
        self.stop_process()
        self.thread = VideoThread()
        self.thread.set_source(0)
        self.connect_thread()
        self.pbar.hide()
        self.thread.start()

    def start_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.jpeg)")
        if file_path:
            self.stop_process()
            self.thread = VideoThread()
            normalized_path = os.path.normpath(file_path)
            self.thread.set_source(normalized_path)
            self.connect_thread()
            self.pbar.show()
            self.pbar.setValue(0)
            self.thread.start()

    def connect_thread(self):
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.progress_signal.connect(self.pbar.setValue)
        self.thread.count_signal.connect(self.update_count)
        self.thread.score_signal.connect(self.update_score)

    def update_image(self, qt_img):
        w = self.image_label.width()
        h = self.image_label.height()
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def update_count(self, count):
        self.info_label.setText(f"Jumlah Lubang: {count}")
    
    def update_score(self, score):
        self.score_label.setText(f"Total Skor: {score}")

    def stop_process(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop() 
            try:
                self.thread.change_pixmap_signal.disconnect()
                self.thread.progress_signal.disconnect()
                self.thread.count_signal.disconnect()
                self.thread.score_signal.disconnect()
            except: pass
            self.thread.wait() 
        self.thread = None
        self.image_label.clear()
        self.reset_calib_ui()
    
    def reset_calib_ui(self):
        self.is_calibration_mode = True
        self.calibration_points = []
        
        self.btn_calib_corners.setText("1. Klik 4 Sudut")
        self.btn_calib_circle.setText("2. Klik Sisi Lingkaran")
        
        self.panel_m3.hide()
        self.panel_m4.hide()
        
        if self.thread:
            self.thread.matrix = None
            self.thread.ellipse_params = None
            self.thread.is_calibrated = False # Cincin hijau disembunyikan saat mereset kalibrasi

    def start_calib_corners(self):
        self.reset_calib_ui()
        self.calibration_type = 'corners'
        self.btn_calib_corners.setText("M1: Klik 4 Sudut...")
        if self.thread: self.thread.show_crosshair = False

    def start_calib_circle(self):
        self.reset_calib_ui()
        self.calibration_type = 'circle'
        self.btn_calib_circle.setText("M2: Klik Atas->Kanan->Bawah->Kiri")
        if self.thread: self.thread.show_crosshair = True 

    def start_calib_angle(self):
        self.reset_calib_ui()
        self.is_calibration_mode = False 
        self.panel_m3.show() 
        self.update_angle_from_sliders()

    # TRIGGER METODE 4 
    def start_calib_ellipse(self):
        if not self.thread or not self.thread.isRunning():
            print("Nyalakan video terlebih dahulu!")
            return
            
        self.reset_calib_ui()
        self.is_calibration_mode = False 
        self.panel_m4.show() 
        
        w = self.thread.original_width
        h = self.thread.original_height
        
        # Atur rentang slider jarak, misal max separuh dari lebar/tinggi video
        self.slider_top.setRange(10, h)
        self.slider_top.setValue(h // 4)
        
        self.slider_bottom.setRange(10, h)
        self.slider_bottom.setValue(h // 4)
        
        self.slider_right.setRange(10, w)
        self.slider_right.setValue(w // 4)
        
        self.slider_left.setRange(10, w)
        self.slider_left.setValue(w // 4)
        
        self.update_ellipse_from_sliders()

    def get_pixel_position(self, event):
        if self.is_calibration_mode and self.thread:
            x_label = event.position().x()
            y_label = event.position().y()
            w_label = self.image_label.width()
            h_label = self.image_label.height()
            w_video = self.thread.original_width
            h_video = self.thread.original_height
            
            actual_x = int((x_label / w_label) * w_video)
            actual_y = int((y_label / h_label) * h_video)
            
            self.calibration_points.append((actual_x, actual_y))

            if len(self.calibration_points) == 4:
                self.is_calibration_mode = False
                self.btn_calib_corners.setText("1. Klik 4 Sudut")
                self.btn_calib_circle.setText("2. Klik Sisi Lingkaran")
                
                if self.calibration_type == 'corners':
                    self.thread.set_calibration_corners(self.calibration_points)
                elif self.calibration_type == 'circle':
                    self.thread.set_calibration_circle(self.calibration_points)

    def update_angle_from_sliders(self):
        pitch = self.slider_pitch.value()
        yaw = self.slider_yaw.value()
        self.pitch_label.setText(f"Kemiringan Vertikal (Pitch): {pitch}°")
        self.yaw_label.setText(f"Kemiringan Horizontal (Yaw): {yaw}°")
        
        if self.thread is not None and self.thread.isRunning():
            self.thread.set_calibration_by_angle(pitch, yaw)

    def update_ellipse_from_sliders(self):
        d_top = self.slider_top.value()
        d_bottom = self.slider_bottom.value()
        d_right = self.slider_right.value()
        d_left = self.slider_left.value()
        
        self.top_label.setText(f"Jarak Atas: {d_top} px")
        self.bottom_label.setText(f"Jarak Bawah: {d_bottom} px")
        self.right_label.setText(f"Jarak Kanan: {d_right} px")
        self.left_label.setText(f"Jarak Kiri: {d_left} px")
        
        if self.thread is not None and self.thread.isRunning():
            self.thread.ellipse_params = (d_top, d_right, d_bottom, d_left)

    def apply_ellipse_transform(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.apply_calibration_ellipse()
            self.panel_m4.hide()