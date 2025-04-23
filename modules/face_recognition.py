import cv2
import face_recognition
import os
import numpy as np
from pathlib import Path


class FaceRecognition:
    def __init__(self, known_faces_dir='data/known_faces'):
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []

        # Create directory if it doesn't exist
        os.makedirs(self.known_faces_dir, exist_ok=True)

        # Load known faces on initialization
        self.load_known_faces()

    def load_known_faces(self):
        """Load all known faces from the specified directory."""
        self.known_face_encodings = []
        self.known_face_names = []

        # Check if directory exists
        if not os.path.exists(self.known_faces_dir):
            return

        # Loop through each file in the known faces directory
        for filename in os.listdir(self.known_faces_dir):
            # Skip files that are not images or are in the temp directory
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')) or filename.startswith("temp_"):
                continue

            try:
                # Get the full path
                image_path = os.path.join(self.known_faces_dir, filename)

                # Use PIL to load and process the image
                from PIL import Image
                import numpy as np

                # Open with PIL and ensure it's RGB
                pil_image = Image.open(image_path).convert('RGB')

                # Convert to numpy array
                face_image = np.array(pil_image)

                # Get the face encoding
                face_encodings = face_recognition.face_encodings(face_image)

                if len(face_encodings) > 0:
                    # Use the first face found in the image
                    encoding = face_encodings[0]
                    self.known_face_encodings.append(encoding)

                    # Use the filename (without extension) as the person's name
                    name = os.path.splitext(filename)[0]
                    self.known_face_names.append(name)
                    print(f"Successfully loaded face: {name}")
                else:
                    print(f"No faces found in image: {filename}")
            except Exception as e:
                print(f"Error loading face from {filename}: {str(e)}")
                continue
    def recognize_faces(self, frame):
        """Recognize faces in the given frame."""
        if frame is None:
            return [], [], []

        # Check if the frame is valid
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            return [], [], []

        # Make sure the frame is in RGB format
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return [], [], []

        # Resize frame to 1/4 size for faster face recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR (OpenCV) to RGB (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        try:
            # Find all face locations and face encodings
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            face_authorized = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                authorized = False

                # Use the known face with the smallest distance to the new face
                if len(self.known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        authorized = True

                face_names.append(name)
                face_authorized.append(authorized)

            # Scale back up face locations since we scaled down the image
            face_locations = [(top * 4, right * 4, bottom * 4, left * 4)
                              for (top, right, bottom, left) in face_locations]

            return face_locations, face_names, face_authorized

        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
            return [], [], []

    def add_known_face(self, image, name):
        """Add a new known face."""
        try:
            # Ensure the known faces directory exists
            os.makedirs(self.known_faces_dir, exist_ok=True)

            # Save the image to a temporary file first
            temp_dir = os.path.join(self.known_faces_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"{name}_temp.jpg")

            print(f"Saving temporary image to {temp_path}")
            cv2.imwrite(temp_path, image)

            # Load the image using PIL (which face_recognition uses internally)
            from PIL import Image
            pil_image = Image.open(temp_path)

            # Convert to RGB mode
            pil_image = pil_image.convert('RGB')

            # Save as a new file
            final_path = os.path.join(self.known_faces_dir, f"{name}.jpg")
            pil_image.save(final_path)
            print(f"Saved processed image to {final_path}")

            # Now let's try to load and process with face_recognition
            try:
                # Load directly using face_recognition's function which is PIL-based
                face_image = face_recognition.load_image_file(final_path)

                # Try to get face encodings
                face_encodings = face_recognition.face_encodings(face_image)

                if len(face_encodings) > 0:
                    print(f"Successfully detected face in image for {name}")
                    # Reload all known faces
                    self.load_known_faces()
                    return True
                else:
                    print(f"No face detected in the processed image for {name}")

                    # Try detecting faces with OpenCV as a fallback
                    try:
                        face_cascade = cv2.CascadeClassifier(
                            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        img = cv2.imread(final_path)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                        if len(faces) > 0:
                            print(f"OpenCV detected {len(faces)} faces in the image")
                            # We'll keep the image since OpenCV found a face
                            self.load_known_faces()
                            return True
                        else:
                            # Remove the image since no face was detected
                            os.remove(final_path)
                            print(f"Removed image at {final_path} as no face was detected")
                            return False
                    except Exception as e:
                        print(f"Error with OpenCV face detection: {str(e)}")
                        return False
            except Exception as e:
                print(f"Error processing with face_recognition: {str(e)}")
                # Try to clean up
                if os.path.exists(final_path):
                    os.remove(final_path)
                return False

        except Exception as e:
            print(f"Exception in add_known_face: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    def draw_faces(self, frame, face_locations, face_names, face_authorized):
        """Draw rectangles and labels for faces on the frame."""
        for (top, right, bottom, left), name, authorized in zip(face_locations, face_names, face_authorized):
            # Draw a rectangle around the face
            color = (0, 255, 0) if authorized else (0, 0, 255)  # Green for authorized, Red for unauthorized
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw a label with the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            status = "Authorized" if authorized else "Unauthorized"
            label = f"{name} ({status})"
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

        return frame