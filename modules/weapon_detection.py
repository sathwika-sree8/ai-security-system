import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO


class WeaponDetection:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold

        # Initialize YOLOv5 model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Use the recommended YOLOv5su model
            try:
                self.model = YOLO('yolov8n.pt')
                print("Using YOLOv5su model as recommended for better performance")
            except Exception as e:
                print(f"Could not load YOLOv5su model: {str(e)}. Falling back to YOLOv5s.")
                self.model = YOLO("yolov5s.pt")

        # Classes for weapon detection
        # YOLOv5 COCO classes that correspond to weapons
        self.weapon_classes = [43, 44, 45, 46, 74]  # knife, scissors, teddy bear, hair drier, gun
    def detect_weapons(self, frame):
        """Detect weapons in the given frame."""
        # Run the detection
        results = self.model(frame)

        # Initialize weapon detection flag
        weapon_detected = False
        detections = []

        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()

                # Check if the detected object is a weapon and confidence is above threshold
                if cls_id in self.weapon_classes and conf > self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2, cls_id, conf))
                    weapon_detected = True

        return weapon_detected, detections

    def draw_detections(self, frame, detections):
        """Draw bounding boxes for detected weapons."""
        for x1, y1, x2, y2, cls_id, conf in detections:
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (147, 112, 219), 2)

            # Get class name
            cls_name = self.model.names[cls_id]

            # Draw label
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame