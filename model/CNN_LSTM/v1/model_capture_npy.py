import csv
from enum import Enum
import os
import queue
import threading
import time
import cv2 
import numpy as np
import mediapipe as mp 



class ProgramShortcuts(Enum):
    quit = ord(u"q")
    start = ord(u"r")
    stop = ord(u"s")

# Initialize variables
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistics = mp.solutions.holistic
holistics = mp_holistics.Holistic(static_image_mode=False, min_detection_confidence=0.65, min_tracking_confidence=0.75)
landmark_seq = []
# will do: mom, dad, bed, iloveyou✅, and potty (bad fist, twist twice) ✅
gesture_action = "NONE" # change this
output_dir = "data/data_3"  # Ensure this directory exists

def mediapipe_detection(image: cv2.typing.MatLike, model):
    return image, model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def draw_landmarks(image, model) -> None:  
    # mp_drawing.draw_landmarks(image, model.face_landmarks, mp_holistics.FACEMESH_CONTOURS,
    #                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=1),
    #                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=1)
    #                         )
    if model.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, model.pose_landmarks, mp_holistics.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
    if model.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, model.right_hand_landmarks, mp_holistics.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=mp_drawing.GREEN_COLOR, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(181, 135, 230), thickness=2, circle_radius=2)
        )
    if model.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, model.left_hand_landmarks, mp_holistics.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=mp_drawing.RED_COLOR, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(181, 135, 230), thickness=2, circle_radius=2)
        )
                       
def extract_keypoints(results):
    # Process pose landmarks (if available)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, pose, lh, rh])

def countdown(seconds):
    for i in range(seconds, 0, -1):
        print(f"Recording starts in {i} seconds...")
        time.sleep(1)
    print("Recording started!")

def save_to_npy(gesture_action, landmark_seq, frame_rate, frame_width, frame_height):
    cur_time = int(time.time_ns())
    output_file = os.path.join(output_dir, f"{gesture_action}_{cur_time}.npy")

    header = (
            [f'{coord}_{i}' for i in range(468) for coord in ('hx', 'hy', 'hz')]+
            [f'{coord}_{i}' for i in range(33) for coord in ('px', 'py', 'pz', "pose_visibility")]+
            [f'{coord}_{i}' for i in range(21) for coord in ('lx', 'ly', 'lz')]+
            [f'{coord}_{i}' for i in range(21) for coord in ('rx', 'ry', 'rz')]+
            ["frame_rate", "frame_width", "frame_height", "frame"]
    )

    # Convert header to a structured array dtype
    dtype = [(name, 'f4') for name in header]
    
    # Initialize a structured array
    num_frames = len(landmark_seq)
    data = np.zeros(num_frames, dtype=dtype)
    
    # Populate the structured array
    for i, frame_data in enumerate(landmark_seq):
        row = np.concatenate([frame_data, np.array([frame_rate, frame_width, frame_height, i])])
        data[i] = tuple(row)
    
    # Save the structured array to a NumPy file
    np.save(output_file, data)

# adjust the values here to get more or less repeats
def auto_capture():
    start_auto_capture(
        num_repeats=68,
        countdown_sec=2,
        capture_duration=1,
        gesture_action=gesture_action,
        frame_rate=frame_rate,
        frame_width=frame_width,
        frame_height=frame_height
    )
def process_frame(frame, results_queue):
    image, results = mediapipe_detection(frame, holistics)
    results_queue.put((image, results))

def start_auto_capture(num_repeats, countdown_sec, capture_duration, gesture_action, frame_rate, frame_width, frame_height):
    global landmark_seq    
    for n in range(num_repeats):
        countdown(countdown_sec)
        landmark_seq = []  # Reset the landmark sequence
        start_time = time.time()

        for _ in range(int(frame_rate * capture_duration)):
            ret, frame = cap.read()
            if not ret:
                break
            
            image, results = mediapipe_detection(frame, holistics)
            draw_landmarks(image, results)
            holistic_landmarks = extract_keypoints(results)
            landmark_seq.append(holistic_landmarks)
            cv2.imshow("Recording Gestures", image)

        print("Recording stopped.")
        save_to_npy(gesture_action, landmark_seq, frame_rate, frame_width, frame_height)
        print(f"total time: {time.time() - start_time}|| count: {n}")
    print("finished")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    key = cv2.waitKey(1) & 0xFF
    match key:
        case ProgramShortcuts.quit.value:
            break
        case ProgramShortcuts.start.value:
            gesture_action = input("please enter the gesture name:")
            threading.Thread(target=auto_capture).start()

    image, results = mediapipe_detection(frame, holistics)
    draw_landmarks(image, results)
    cv2.imshow("Recording Gestures", image)
    

cap.release()
cv2.destroyAllWindows()