# 🛡️ AI Security System

An intelligent, real-time security solution that detects **faces**, **weapons**, and **aggression** in video streams.  
Built using **Python**, **OpenCV**, **Deep Learning**, and a **Streamlit** interface for easy access to logs and tools.  

![Weapon Detection Example]("C:\Users\SATHWIKA\OneDrive\Desktop\ai security system\images\Screenshot 2025-05-21 134253.png")


---

## ✨ Features

### 👤 Face Recognition
- Detects and identifies known faces from video feed.
- Unauthorized individuals are marked with red bounding boxes.
- Uses `face_recognition` and `OpenCV`.

### 🔫 Weapon Detection
- Detects weapons in live video streams.
- Captures frames with weapons into the `captured_images/` folder.
- Logs the **start and end time** of weapon detection events into an **SQLite** database.
- View logs with date filters in the **Security Logs** section of the Streamlit app.

### 💢 Aggression Detection
- Upload any video via the Streamlit interface.
- Classifies the video as **Fight Detected** or **No Fight**.
- Allows users to **download the analyzed video** directly from the app.

---

## 💻 Tech Stack

| Component             | Tools / Libraries                          |
|----------------------|---------------------------------------------|
| Language             | Python                                      |
| Face Recognition     | `face_recognition`, `OpenCV`                |
| Weapon Detection     | Custom Model (YOLO / CNN-based)             |
| Aggression Detection | Deep Learning, Video Classification         |
| Web UI               | `Streamlit`                                 |
| Database             | `SQLite`                                    |

---

## 🗂️ Folder Structure

ai-security-system/
├── app.py # Core backend logic
├── face_recog.py # Face recognition module
├── captured_images/ # Saved frames with weapons
├── aggression_detection.py # Fight detection from uploaded video
├── streamlit_app.py # UI and user interface logic
├── database.db # SQLite DB with detection logs
├── data/
│ └── known_faces/ # Folder with images of known individuals
├── requirements.txt
└── README.md

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/sathwika-sree8/ai-security-system.git
cd ai-security-system

python -m venv .venv
.venv\Scripts\activate  # On Windows
pip install -r requirements.txt

streamlit run app.py

