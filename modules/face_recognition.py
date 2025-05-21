import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN

class FaceRecognition:
    def __init__(self):
        self.embedder = FaceNet()
        self.known_faces = {}  # {name: embedding}
        self.detector = MTCNN()

    def get_face_embedding(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb_image)

        if len(detections) == 0:
            return None, None

        embeddings = []
        face_locations = []

        for det in detections:
            x, y, w, h = det['box']
            x, y = max(0, x), max(0, y)
            face = image[y:y+h, x:x+w]

            if face.shape[0] < 50 or face.shape[1] < 50:
                continue  # Ignore too small faces

            face = cv2.resize(face, (160, 160))
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            embedding = self.embedder.embeddings([face_rgb])[0]

            embeddings.append(embedding)
            face_locations.append((y, x + w, y + h, x))  # (top, right, bottom, left)

        return face_locations, embeddings

    def add_known_face(self, name, image):
        face_locations, embeddings = self.get_face_embedding(image)
        if face_locations and embeddings:
            self.known_faces.setdefault(name, []).append(embeddings)
            # Assuming one face per image
            return True
        return False

    def recognize_faces(self, image, threshold=0.6):
        face_locations, embeddings = self.get_face_embedding(image)
        if not embeddings:
            return [], [], []

        face_names = []
        face_authorized = []

        for embedding in embeddings:
            identity = "Unknown"
            authorized = False
            min_dist = float('inf')

            for name, embeddings_list in self.known_faces.items():
                for known_embedding in embeddings_list:
                    dist = np.linalg.norm(embedding - known_embedding)
                    if dist < min_dist and dist < threshold:
                        min_dist = dist
                        identity = name
                        authorized = True

            face_names.append(identity)
            face_authorized.append(authorized)

        return face_locations, face_names, face_authorized
