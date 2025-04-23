import cv2
import os
import datetime
import numpy as np


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


def resize_frame(frame, width=None, height=None):
    """Resize a frame to the specified dimensions."""
    if width is None and height is None:
        return frame

    if width is None:
        ratio = height / frame.shape[0]
        width = int(frame.shape[1] * ratio)
    elif height is None:
        ratio = width / frame.shape[1]
        height = int(frame.shape[0] * ratio)

    resized_frame = cv2.resize(frame, (width, height))
    return resized_frame


def convert_to_rgb(frame):
    """Convert a frame from BGR to RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)