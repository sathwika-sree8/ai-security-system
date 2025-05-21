# motion_detection.py
import cv2

class MotionDetector:
    def __init__(self):
        self.prev_frame = None

    def detect_motion(self, frame):
        # Convert frame to grayscale

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray_frame
            return False, frame  # No motion detected initially

        # Compute absolute difference between the current frame and previous frame
        frame_diff = cv2.absdiff(self.prev_frame, gray_frame)

        # Threshold the difference to get a binary image
        _, thresh_frame = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Find contours to identify areas of motion
        contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) > 5000:  # Minimum area for detecting significant motion
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                motion_detected = True

        self.prev_frame = gray_frame  # Update previous frame

        return motion_detected, frame
