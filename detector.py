import cv2
from ultralytics import YOLO

class BulletDetector:
    def __init__(self, model_path='weights/bullet_model.pt'):
        self.model = YOLO(model_path) 

    def get_video_info(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames

    def detect_frame(self, frame):
        results = self.model(frame, conf=0.6, classes=[0, 1], verbose=False)
        
        boxes_data = []
        # Ambil hasil deteksi mentah
        for box in results[0].boxes:
            cls = int(box.cls[0])
            xywh = box.xywh[0].tolist() # [x_center, y_center, w, h]
            boxes_data.append((cls, xywh[0], xywh[1], xywh[2], xywh[3]))
                
        annotated_frame = results[0].plot(line_width=2, font_size=0.5, labels=False)
        return annotated_frame, boxes_data

# Script Testing Standalone (untuk mencoba tanpa GUI)
if __name__ == "__main__":
    # Gunakan webcam (0) atau ganti dengan path video 'assets/video.mp4'
    source = 0 
    
    # Inisialisasi detector
    detector = BulletDetector('weights/bullet_model.pt')
    
    cap = cv2.VideoCapture(source)
    print("Menjalankan Test Deteksi... Tekan 'q' untuk keluar.")

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Proses frame
            img_result, total = detector.detect_frame(frame)
            
            # Tambahkan info jumlah di pojok kiri atas
            cv2.putText(img_result, f"Bullet Holes: {total}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Tampilkan hasil
            cv2.imshow("Detector Test", img_result)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()