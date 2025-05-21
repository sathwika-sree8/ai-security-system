import os
import cv2
import face_recognition
import numpy as np

class FaceRecognition:
    def __init__(self, known_faces_dir='data/known_faces', tolerance=0.5):
        """
        Initialize FaceRecognition class by loading known faces.

        Args:
            known_faces_dir (str): Directory path with known face images.
            tolerance (float): Distance tolerance for face matching.
        """
        self.known_encodings = []
        self.known_names = []
        self.tolerance = tolerance
        self.load_known_faces(known_faces_dir)

    def load_known_faces(self, known_faces_dir):
        """
        Load known faces and their encodings from the specified directory.
        """
        for filename in os.listdir(known_faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_encodings.append(encodings[0])
                    self.known_names.append(os.path.splitext(filename)[0])
                else:
                    print(f"No face found in {filename}, skipping.")

    def recognize_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_authorized = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=self.tolerance)
            name = "Unauthorized"
            authorized = False

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_names[first_match_index]
                authorized = True

            face_names.append(name)
            face_authorized.append(authorized)

            # Scale back up
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if name != "Unauthorized" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return face_locations, face_names, face_authorized  # ✅ Returns 3 values
