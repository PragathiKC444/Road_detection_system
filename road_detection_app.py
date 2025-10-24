import sys
import cv2
import numpy as np
import os
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget


class RoadDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Road Detection (People, Vehicles)')
        self.setGeometry(100, 100, 800, 650)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create a label to display the video stream
        self.label = QLabel(self)
        self.label.resize(800, 600)
        self.layout.addWidget(self.label)

        # Create buttons for control
        self.start_button = QPushButton("Start Detection")
        self.stop_button = QPushButton("Stop Detection")
        self.quit_button = QPushButton("Quit")
        
        # Toggle buttons for features
        self.toggle_road_button = QPushButton("Toggle Road Detection")
        self.toggle_pedestrian_button = QPushButton("Toggle Pedestrian Detection")
        self.toggle_vehicle_button = QPushButton("Toggle Vehicle Detection")
        
        # Snapshot button
        self.snap_button = QPushButton("Capture Snapshot")

        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        self.quit_button.clicked.connect(self.close_application)
        self.toggle_road_button.clicked.connect(self.toggle_road_detection)
        self.toggle_pedestrian_button.clicked.connect(self.toggle_pedestrian_detection)
        self.toggle_vehicle_button.clicked.connect(self.toggle_vehicle_detection)
        self.snap_button.clicked.connect(self.capture_snapshot)

        # Add buttons to the layout
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.stop_button)
        self.layout.addWidget(self.quit_button)
        self.layout.addWidget(self.toggle_road_button)
        self.layout.addWidget(self.toggle_pedestrian_button)
        self.layout.addWidget(self.toggle_vehicle_button)
        self.layout.addWidget(self.snap_button)

        # Status label
        self.status_label = QLabel("Status: All detections OFF", self)
        self.layout.addWidget(self.status_label)

        # Set up HOG descriptor for detecting pedestrians
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Initialize the webcam feed (0 for default camera)
        self.cap = cv2.VideoCapture(0)

        # Set up a background subtractor for motion detection (vehicles, moving objects)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

        # Set up a timer to periodically update the frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Initially, detection is stopped
        self.is_running = False
        self.detect_road = False
        self.detect_pedestrian = False
        self.detect_vehicle = False

        # Create the 'snapshots' directory if it doesn't exist
        if not os.path.exists("snapshots"):
            os.makedirs("snapshots")

    def start_detection(self):
        if not self.is_running:
            self.is_running = True
            self.timer.start(20)  # Update every 20ms (~50fps)
            self.status_label.setText("Status: Detection ON")

    def stop_detection(self):
        if self.is_running:
            self.is_running = False
            self.timer.stop()
            self.status_label.setText("Status: All detections OFF")

    def close_application(self):
        self.stop_detection()
        self.close()

    def toggle_road_detection(self):
        self.detect_road = not self.detect_road
        if self.detect_road:
            self.status_label.setText("Status: Road Detection ON")
        else:
            self.status_label.setText("Status: Road Detection OFF")

    def toggle_pedestrian_detection(self):
        self.detect_pedestrian = not self.detect_pedestrian
        if self.detect_pedestrian:
            self.status_label.setText("Status: Pedestrian Detection ON")
        else:
            self.status_label.setText("Status: Pedestrian Detection OFF")

    def toggle_vehicle_detection(self):
        self.detect_vehicle = not self.detect_vehicle
        if self.detect_vehicle:
            self.status_label.setText("Status: Vehicle Detection ON")
        else:
            self.status_label.setText("Status: Vehicle Detection OFF")

    def capture_snapshot(self):
        # Capture the current frame
        ret, frame = self.cap.read()
        if ret:
            # Save the snapshot with a unique name
            snapshot_filename = f"snapshots/snapshot_{cv2.getTickCount()}.png"
            cv2.imwrite(snapshot_filename, frame)
            self.status_label.setText(f"Snapshot saved: {snapshot_filename}")

    def update_frame(self):
        if not self.is_running:
            return

        # Capture the frame from the webcam
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert the frame to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pedestrian detection
        if self.detect_pedestrian:
            boxes, _ = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Vehicle detection using background subtraction
        if self.detect_vehicle:
            fg_mask = self.bg_subtractor.apply(frame)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 1000:  # Filter small noise
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue boxes for vehicles/moving objects

        # Road detection: Edge detection and Hough transform for lane markings
        if self.detect_road:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Define the region of interest (ROI) where roads are typically located
            height, width = edges.shape
            roi = edges[int(height/2):, :]  # Only consider the lower half of the frame for road detection

            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=150)

            # Draw detected road lines
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Draw lines in green (road boundaries)
                    cv2.line(frame, (x1, y1 + int(height/2)), (x2, y2 + int(height/2)), (0, 255, 0), 2)

        # Convert the frame to RGB format for displaying in PyQt
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Set the QImage as the pixmap for the label
        self.label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


# Main execution
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RoadDetectionApp()
    window.show()
    sys.exit(app.exec_())