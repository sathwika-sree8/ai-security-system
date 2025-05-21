import streamlit as st
import cv2
import tempfile
import os
import time
import numpy as np
from PIL import Image
from pathlib import Path
import datetime
import atexit
from modules.face_recognition import FaceRecognition
from modules.motion_detection import MotionDetector
from modules.aggression_detector import detect_aggression

# Import only the essential modules
from modules.weapon_detection import WeaponDetection
from modules.database import SecurityDatabase

# Set page title and layout
st.set_page_config(page_title="AI Security System", layout="wide")

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'detected_weapons' not in st.session_state:
    st.session_state.detected_weapons = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None

# Initialize modules
weapon_detector = WeaponDetection()
db = SecurityDatabase()
# Helper functions
def convert_to_rgb(frame):
    """Convert a frame from BGR to RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_frame(frame, directory="data/captured_frames"):
    """Save a frame to the specified directory."""
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(directory, filename)

    # Save the frame
    cv2.imwrite(filepath, frame)

    return filepath


# Application title
st.title("AI Security System")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Security Monitor", "Security Logs"])

with tab1:
    # Security monitor tab
    st.header("Live Weapon Detection")

    # Camera control buttons
    col1, col2, col3 = st.columns(3)

    if col1.button("Start Camera" if not st.session_state.camera_active else "Stop Camera"):
        st.session_state.camera_active = not st.session_state.camera_active

    # Add camera selection option
    camera_options = ["Default (0)", "Camera 1", "Camera 2", "Camera 3"]
    selected_camera = col2.selectbox("Select Camera", camera_options)
    camera_index = int(selected_camera.split("(")[1].split(")")[0]) if "(" in selected_camera else camera_options.index(
        selected_camera)

    # Debug option
    show_debug = col3.checkbox("Show Debug Info")

    # Placeholder for video feed
    video_placeholder = st.empty()
    debug_placeholder = st.empty()

    # Status indicator
    weapon_status = st.empty()

    # Camera feed loop
    if st.session_state.camera_active:
        # Initialize camera with selected index
        cap = cv2.VideoCapture(camera_index)

        # Try to set camera properties for better compatibility
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            st.error(
                f"Error: Could not open camera with index {camera_index}. Please try a different camera or check if it's being used by another application.")
            st.session_state.camera_active = False
        else:
            # Display camera properties if debug is enabled
            if show_debug:
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                debug_placeholder.info(f"Camera properties: Width={width}, Height={height}, FPS={fps}")

            # Counter for frame reading attempts
            read_attempts = 0
            max_attempts = 5

            while st.session_state.camera_active:
                # Read frame from camera
                ret, frame = cap.read()

                if not ret or frame is None or frame.size == 0:
                    read_attempts += 1
                    if read_attempts >= max_attempts:
                        st.error("Error: Failed to capture frames from camera after multiple attempts.")
                        st.info(
                            "Try selecting a different camera from the dropdown or check if your camera is working properly.")
                        st.session_state.camera_active = False
                        break

                    if show_debug:
                        debug_placeholder.warning(f"Failed to read frame, attempt {read_attempts}/{max_attempts}")

                    time.sleep(1)  # Wait before retrying
                    continue

                # Reset attempt counter on successful frame
                read_attempts = 0

                # Check if frame is valid
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    if show_debug:
                        debug_placeholder.warning(f"Invalid frame format: shape={frame.shape}")
                    continue

                # Store current frame
                st.session_state.current_frame = frame.copy()

                try:
                    # Weapon detection
                    weapon_detected, detections = weapon_detector.detect_weapons(frame)

                    # Update session state
                    st.session_state.detected_weapons = weapon_detected

                    # Draw weapon detections
                    if weapon_detected:
                        frame = weapon_detector.draw_detections(frame, detections)

                    # Display frame
                    video_placeholder.image(convert_to_rgb(frame), channels="RGB", use_container_width=True)

                    # Update status indicators
                    if weapon_detected:
                        weapon_status.error("⚠️ Weapon detected!")

                        # Save frame and log security event
                        frame_path = save_frame(frame)
                        db.add_log(
                            face_id="Unknown",
                            is_authorized=False,
                            weapon_detected=True,
                            image_path=frame_path
                        )
                    else:
                        weapon_status.success("✅ No weapons detected")

                except Exception as e:
                    if show_debug:
                        debug_placeholder.error(f"Error processing frame: {str(e)}")

                # Add small delay to reduce CPU usage
                time.sleep(0.1)

            # Release camera when stopped
            cap.release()

with tab2:
    # Security logs tab
    st.header("Security Logs")

    # Date range selector
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date")
    end_date = col2.date_input("End Date")

    # Format dates for SQL query
    start_datetime = f"{start_date} 00:00:00"
    end_datetime = f"{end_date} 23:59:59"

    # Get logs within date range
    logs = db.get_logs_by_date(start_datetime, end_datetime)

    # Display logs
    if logs:
        st.write(f"Found {len(logs)} security events")

        # Create a table of logs
        log_data = []
        for log in logs:
            weapon = "Yes" if log['weapon_detected'] else "No"
            log_data.append({
                "Time": log['timestamp'],
                "Weapon Detected": weapon
            })

        st.table(log_data)
    else:
        st.info("No security events found in the selected date range")

# Add a footer
# Face detection toggle
# Face detection toggle and camera handling section
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔍 Face Detection")

if "face_detection_active" not in st.session_state:
    st.session_state.face_detection_active = False

# Face detection button placed at the top with other camera buttons
if col1.button("Face Detection"):
    st.session_state.face_detection_active = not st.session_state.face_detection_active

# Placeholder for face feed
face_frame_placeholder = st.empty()

# Initialize face recognition system
face_recog = FaceRecognition()


if st.session_state.face_detection_active:
    # Separate camera handling for face detection
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while st.session_state.face_detection_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video feed 😞")
            break

        # Detect faces and get authorization status
        face_locations, face_names, face_authorized = face_recog.recognize_faces(frame)

        for (top, right, bottom, left), name, auth in zip(face_locations, face_names, face_authorized):
            label = f"{'✅ Authorized' if auth else '❌ Unauthorized'}: {name}"
            color = (0, 255, 0) if auth else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Tiny delay to prevent CPU overuse
        time.sleep(0.05)

    cap.release()
    st.success("Face detection stopped 💖")


menu = st.sidebar.selectbox("Choose Module", ["Home", "Motion Detector"])

if menu == "Motion Detector":
    st.header("🚨 Motion Detection")

    # Checkbox to start or stop the camera
    run = st.checkbox("Start Camera")

    # Initialize the motion detector (you can replace it with your custom PoseDetector if needed)
    motion_detector = MotionDetector()
    FRAME_WINDOW = st.image([])  # Placeholder for displaying frames

    cap = cv2.VideoCapture(0)  # Access the webcam

    if run:
        st.write("Motion detection is ON. Wait for any significant movement...")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera not accessible.")
                break

            # Call the detect_motion function from MotionDetector class
            motion_detected, frame = motion_detector.detect_motion(frame)

            if motion_detected:
                st.error("🚨 Motion Detected!")  # Display message if motion is detected

            # Convert and display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

            time.sleep(0.03)  # Adjust for better performance and camera handling

        cap.release()
    else:
        st.write("Camera is off.")

menu = st.sidebar.selectbox("Choose Module", ["Home", "Aggressive Behavior Recognition"])

# Home Page
if menu == "Home":
    st.title("📊 Welcome to the Smart Surveillance Dashboard")
    st.write("Select a module from the sidebar to get started.")

# Aggression Detection Module
elif menu == "Aggressive Behavior Recognition":
    st.header("🧠 Aggressive Behavior Recognition")

    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
# Checkbox to enable aggression detection
    enable_detection = st.checkbox("Enable Aggressive Behavior Recognition")

    if video_file is not None and enable_detection:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        st.info("Processing video... Please wait ⏳")

    # Run detection from aggression_detector.py
        output_path, status_log = detect_aggression(tfile.name)

        st.video(output_path)

    # Show status messages from detection
        st.subheader("Detection Summary:")
        for status in status_log:
            st.write(status)

    # Download link
        with open(output_path, "rb") as file:
            btn = st.download_button(
                label="Download Annotated Video",
                data=file,
                file_name="annotated_output.mp4",
                mime="video/mp4"
        )

        time.sleep(1)


        def delete_temp_file(file_path):
            os.remove(file_path)

        atexit.register(delete_temp_file, tfile.name)

st.markdown("---")
st.markdown("AI Security System © 2025")
