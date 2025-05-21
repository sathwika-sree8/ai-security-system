import cv2
import os
import tempfile

def detect_aggression(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_path = os.path.join(tempfile.gettempdir(), "output_annotated.mp4")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (
        int(cap.get(3)), int(cap.get(4))))

    status_log = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Fake aggression detection logic (replace with model)
        if frame_num % 100 == 0:
            cv2.putText(frame, "Fight Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            status_log.append(f"Frame {frame_num}: Fight Detected")
        else:
            cv2.putText(frame, "No Aggression", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    return output_path, status_log
