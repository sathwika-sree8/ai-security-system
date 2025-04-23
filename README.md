# AI Security System

A Python-based security system using AI for face recognition and weapon detection.

## Features

- Face detection and recognition using OpenCV
- Classification of faces as authorized or unauthorized
- Weapon detection using YOLOv5
- Security event logging in SQLite database
- Streamlit web interface for monitoring and management

## Requirements

- Python 3.9+
- OpenCV
- Face Recognition library
- YOLOv5
- Streamlit
- SQLite

## Setup

1. Install dependencies:
   pip install -r requirements.txt
2. Run the application:
streamlit run app.py

3. Upload known face images through the "Upload Known Face" tab.
4. Monitor the security feed in the "Security Monitor" tab.
## Project Structure
- `app.py`: Main Streamlit application
- `modules/`: Contains core functionality modules
- `face_recognition.py`: Face detection and recognition logic
- `weapon_detection.py`: YOLOv5 weapon detection logic
- `database.py`: SQLite database operations
- `utils/`: Utility functions
- `data/`: Data storage
- `known_faces/`: Directory for known face images
- `models/`: YOLOv5 model storage
- `security_logs.db`: SQLite database file
Create the Directory Structure
To complete the setup, you'll need to create the directory structure as outlined earlier:

Create the main directory (ai_security_system)
Create subdirectories (modules, utils, data, data/known_faces, data/models)
Add the above files to their respective directories
Once the structure is created and files are in place, you can run the application with:

streamlit run app.py
