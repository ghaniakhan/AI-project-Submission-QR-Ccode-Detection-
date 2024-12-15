import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer, Qt
from pyzbar.pyzbar import decode
from tensorflow.keras.models import load_model
import joblib
from skimage.feature import hog

# Load models
qr_barcode_model = load_model('barcode_qr_model.h5')  # First model
rf_classifier = joblib.load("rf_qr_code_model.pkl")  # Second model
label_encoder = joblib.load("label_encoder.pkl")

class QRScannerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QR/Barcode Scanner")
        self.setGeometry(100, 100, 1000, 700)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Error: Camera not accessible.")

        # Set up UI
        self.init_ui()

        # Timer for video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)  # Update every 20ms

    def init_ui(self):
        # Central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Layouts
        self.main_layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()

        # Video feed label
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.video_label)

        # Data display label
        self.data_label = QLabel("Data: N/A")
        self.data_label.setStyleSheet("font-size: 16px; color: #2c3e50;")
        self.main_layout.addWidget(self.data_label)

        # Classification result label
        self.result_label = QLabel("Classification: N/A")
        self.result_label.setStyleSheet("font-size: 16px; color: #27ae60;")
        self.main_layout.addWidget(self.result_label)

        # Warning label
        self.warning_label = QLabel("Status: No QR detected.")
        self.warning_label.setStyleSheet("font-size: 16px; color: #e74c3c;")
        self.main_layout.addWidget(self.warning_label)

        # Buttons
        self.start_button = QPushButton("Start Camera")
        self.start_button.setStyleSheet("background-color: #3498db; color: white; font-size: 14px; padding: 10px;")
        self.start_button.clicked.connect(self.start_camera)

        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.setStyleSheet("background-color: #e74c3c; color: white; font-size: 14px; padding: 10px;")
        self.stop_button.clicked.connect(self.stop_camera)

        self.exit_button = QPushButton("Exit")
        self.exit_button.setStyleSheet("background-color: #2c3e50; color: white; font-size: 14px; padding: 10px;")
        self.exit_button.clicked.connect(self.close_app)

        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.stop_button)
        self.button_layout.addWidget(self.exit_button)

        self.main_layout.addLayout(self.button_layout)
        self.central_widget.setLayout(self.main_layout)

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return processed

    def check_maliciousness(self, qr_region):
        qr_region_resized = cv2.resize(qr_region, (32, 32))
        fd, _ = hog(qr_region_resized, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True)
        prediction = rf_classifier.predict([fd])
        label = label_encoder.inverse_transform(prediction)[0]
        return label

    def classify_qr(self, frame):
        resized_frame = cv2.resize(frame, (224, 224))
        image = np.expand_dims(resized_frame, axis=0)
        image = image / 255.0
        predictions = qr_barcode_model.predict(image)
        code_type = np.argmax(predictions)  # Assuming multi-class classification
        return code_type

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Preprocess and detect QR codes
        preprocessed_frame = self.preprocess_frame(frame)
        decoded_objects = decode(preprocessed_frame)

        for obj in decoded_objects:
            points = obj.polygon
            if points:
                points = [(int(p.x), int(p.y)) for p in points]
                data = obj.data.decode('utf-8')

                # Highlight QR code
                cv2.polylines(frame, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=3)

                # Region of interest
                x_min = max(0, min(p[0] for p in points))
                y_min = max(0, min(p[1] for p in points))
                x_max = min(frame.shape[1], max(p[0] for p in points))
                y_max = min(frame.shape[0], max(p[1] for p in points))

                qr_region = cv2.cvtColor(frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)

                # Maliciousness check
                label = self.check_maliciousness(qr_region)
                self.data_label.setText(f"Data: {data}")
                self.result_label.setText(f"Classification: {label}")
                self.warning_label.setText(f"Status: {'Malicious' if label == 'malicious' else 'Safe'}")

        # Convert frame for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def start_camera(self):
        self.timer.start(20)

    def stop_camera(self):
        self.timer.stop()

    def close_app(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = QRScannerApp()
    main_window.show()
    sys.exit(app.exec_())
