import os
import sys
import cv2
import math
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QProgressBar)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from detector import BulletDetector

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
    
    def set_calibration(self, points):
        # Urutan: TL, TR, BR, BL
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])
        self.matrix = cv2.getPerspectiveTransform(pts1, pts2)

    def set_source(self, source):
        self.source = source

    def calculate_score_ellipse(self, hole_center, target_center, r_h, r_v):
        if r_h <= 0 or r_v <= 0: return 0

        dx = hole_center[0] - target_center[0]
        dy = hole_center[1] - target_center[1]
        
        # Jarak normalisasi elips
        # Nilai 1.0 berarti tepat berada di garis pinggir r_h dan r_v (batas skor 9)
        normalized_dist = math.sqrt((dx**2 / r_h**2) + (dy**2 / r_v**2))
        
        # Karena area hitam meliputi skor 9 dan 10, batas lebarnya adalah 0.5 per zona
        if normalized_dist <= 0.5:
            return 10
        elif normalized_dist <= 1.0:
            return 9
        else:
            # Area putih (skor 8 ke bawah)
            score = 8 - int((normalized_dist - 1.0) / 0.5)
            return max(0, score)

    def run(self):
        cap = cv2.VideoCapture(self.source)
        self.original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        is_video = isinstance(self.source, str)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 0

        while self._run_flag:
            ret, frame = cap.read()
            if not ret or not self._run_flag: break

            try:
                # A. Warping
                if self.matrix is not None:
                    working_frame = cv2.warpPerspective(frame, self.matrix, (500, 500))
                else:
                    working_frame = frame

                # B. Deteksi
                annotated_frame, boxes_data = self.detector.detect_frame(working_frame)
                
                # C. Olah Data Deteksi
                hole_centers = []
                target_center = None
                target_r_width = 0
                target_r_height = 0

                for cls, x, y, w, h in boxes_data:
                    if cls == 1: # bullet_hole
                        hole_centers.append((x, y))
                    elif cls == 0: # black_contour
                        target_center = (x, y)
                        target_r_width = (w / 2) * 1.3
                        target_r_height = (h / 2) * 1.3

                # D. Hitung Skor & Gambar Lingkaran Hijau (Elips)
                current_total_score = 0
                if target_center is not None and target_r_width > 0 and target_r_height > 0:
                    
                    # 1. Gambar 10 Jalur Visual dengan cv2.ellipse
                    for i in range(1, 11):
                        # Ukuran (width, height) untuk masing-masing zona
                        axes = (int(target_r_width * (i/2)), int(target_r_height * (i/2)))
                        
                        cv2.ellipse(annotated_frame, 
                                    (int(target_center[0]), int(target_center[1])), 
                                    axes, 0, 0, 360, (0, 255, 0), 1)

                    # 2. Hitung skor per lubang dengan rumus elips
                    for hole in hole_centers:
                        current_total_score += self.calculate_score_ellipse(
                            hole, target_center, target_r_width, target_r_height
                        )

                # E. Emit Signal
                self.count_signal.emit(len(hole_centers))
                self.score_signal.emit(current_total_score)
                
                # ... Konversi QImage & Emit (Gunakan .copy() seperti sebelumnya) ...
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
        self.setMinimumSize(900, 700)
        
        # Main Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # 1. Layar Video
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.image_label.setFixedSize(850, 480)
        self.main_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 2. Progress Bar
        self.pbar = QProgressBar(self)
        self.pbar.setValue(0)
        self.main_layout.addWidget(self.pbar)

        # 3. Info Panel
        self.info_layout = QHBoxLayout()
        self.info_label = QLabel("Jumlah Lubang Peluru: 0", self)
        self.score_label = QLabel("Total Skor: 0", self)

        label_style = "font-size: 18px; font-weight: bold; color: #ffffff; padding: 10px;"
        self.info_label.setStyleSheet(label_style)
        self.score_label.setStyleSheet(label_style)

        self.info_layout.addWidget(self.info_label)
        self.info_layout.addWidget(self.score_label)
        self.main_layout.addLayout(self.info_layout)

        # 4. Tombol Kontrol
        self.btn_layout = QHBoxLayout()
        
        self.btn_webcam = QPushButton("Gunakan Webcam")
        self.btn_webcam.clicked.connect(self.start_webcam)
        
        self.btn_video = QPushButton("Pilih File Video")
        self.btn_video.clicked.connect(self.start_video)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_process)
        
        self.btn_layout.addWidget(self.btn_webcam)
        self.btn_layout.addWidget(self.btn_video)
        self.btn_layout.addWidget(self.btn_stop)
        self.main_layout.addLayout(self.btn_layout)

        # Init Thread
        self.thread = None

        self.calibration_points = [] # Menyimpan 4 titik (x, y)
        self.is_calibration_mode = False
        
        # Tambahkan tombol Kalibrasi di UI
        self.btn_calibrate = QPushButton("Mulai Kalibrasi")
        self.btn_calibrate.clicked.connect(self.toggle_calibration)
        self.btn_layout.addWidget(self.btn_calibrate)

        # Override fungsi klik pada label
        self.image_label.mousePressEvent = self.get_pixel_position

    def start_webcam(self):
        self.stop_process()
        self.thread = VideoThread()
        self.thread.set_source(0)
        self.connect_thread()
        self.pbar.hide() # Sembunyikan progress bar untuk webcam
        self.thread.start()

    def start_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        
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
        # Ambil ukuran label saat ini
        w = self.image_label.width()
        h = self.image_label.height()
        
        # Scale pixmap agar muat di dalam label tanpa merusak rasio (KeepAspectRatio)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(
            w, h, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
    
        self.image_label.setPixmap(scaled_pixmap)

    def update_count(self, count):
        self.info_label.setText(f"Jumlah Lubang Terdeteksi: {count}")
    
    def update_score(self, score):
        self.score_label.setText(f"Total Skor: {score}")

    def stop_process(self):
        if self.thread is not None and self.thread.isRunning():
            # 1. Beritahu loop di dalam thread untuk berhenti
            self.thread.stop() 
            
            # 2. Putuskan koneksi signal agar tidak ada update UI saat thread sedang sekarat
            try:
                self.thread.change_pixmap_signal.disconnect()
                self.thread.progress_signal.disconnect()
                self.thread.count_signal.connect(self.update_count)
                self.thread.score_signal.disconnect()
            except:
                pass # Jika sudah terputus, abaikan

            # 3. Tunggu sampai thread benar-benar mati sebelum menghapus objeknya
            self.thread.wait() 
            
        self.thread = None
        self.image_label.clear()
    
    def toggle_calibration(self):
        self.is_calibration_mode = True
        self.calibration_points = []
        self.btn_calibrate.setText("Klik 4 Sudut Target...")
        self.btn_calibrate.setEnabled(False)

    def get_pixel_position(self, event):
        if self.is_calibration_mode and self.thread:
            # 1. Ambil posisi klik di Label
            x_label = event.position().x()
            y_label = event.position().y()
            
            # 2. Ambil ukuran Label saat ini
            w_label = self.image_label.width()
            h_label = self.image_label.height()
            
            # 3. Ambil resolusi asli video dari thread
            # Pastikan di VideoThread kamu sudah menyimpan self.original_width & height
            w_video = self.thread.original_width
            h_video = self.thread.original_height
            
            # 4. KONVERSI SKALA (PENTING!)
            # Menghitung posisi koordinat yang sebenarnya pada file video
            actual_x = int((x_label / w_label) * w_video)
            actual_y = int((y_label / h_label) * h_video)
            
            self.calibration_points.append((actual_x, actual_y))
            print(f"Titik {len(self.calibration_points)} (Resolusi Video): {actual_x}, {actual_y}")

            if len(self.calibration_points) == 4:
                self.is_calibration_mode = False
                self.btn_calibrate.setText("Kalibrasi Selesai")
                self.btn_calibrate.setEnabled(True)
                
                # Kirim koordinat asli ke thread
                self.thread.set_calibration(self.calibration_points)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec())