from enum import Enum
import time
from typing import Counter
import joblib
import mediapipe as mp
import cv2
import numpy as np 
from keras import models
import pandas as pd

class ProgramShortcuts(Enum):
    quit = ord(u"q")
    start = ord(u"r")
    stop = ord(u"s")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistics = mp.solutions.holistic
holistics = mp_holistics.Holistic(static_image_mode=False, min_detection_confidence=0.65, min_tracking_confidence=0.8)

model = models.load_model(r"model\CNN_LSTM\v3\models\model_11_v3_1724733396947051100.keras")
class_labels = pd.read_csv(r"model\CNN_LSTM\v3\class_labels.csv")["gesture"].tolist()
preprocessor = joblib.load(r"model\CNN_LSTM\v3\preprocessing_pipeline.pkl")

def predict(landmark_seq, frame_rate, frame_width, frame_height, gesture_action=""):
    header = (
            [f'{coord}_{i}' for i in range(468) for coord in ('hx', 'hy', 'hz')]+
            [f'{coord}_{i}' for i in range(33) for coord in ('px', 'py', 'pz', "pose_visibility")]+
            [f'{coord}_{i}' for i in range(21) for coord in ('lx', 'ly', 'lz')]+
            [f'{coord}_{i}' for i in range(21) for coord in ('rx', 'ry', 'rz')]+
            ["frame_rate", "frame_width", "frame_height", "frame", "gesture_index"]
        )
    
    data = [
        frame_data + [frame_rate, frame_width, frame_height, i, time.time_ns()]  for i, frame_data in enumerate(landmark_seq)
    ]

    df = pd.DataFrame(data, columns=header)
    
    landmark_columns = [f"{col}" for col in df.columns if col.startswith(("hx", "hy", "hz", "px", "py", "pz", "lx", "ly", "lz", "rx", "ry", "rz"))]
    categorical_columns = ["gesture_index"]
    numerical_columns = ["frame", "frame_rate", "frame_width", "frame_height"] + [f"{col}" for col in df.columns if col.startswith("pose_visibility")]

    X_new_pre = preprocessor.transform(df[landmark_columns + numerical_columns + categorical_columns])
    X_new_pre.drop(columns="remainder__gesture_index", inplace=True)
    
    X_new = np.reshape(X_new_pre, (1, X_new_pre.shape[0], X_new_pre.shape[1]))
    pred = model.predict(X_new, verbose=0)

    pred_labels = [class_labels[np.argmax(p)] for p in pred]
    formatted_predictions = [
        f"{class_labels[i]}: {prob:.2f}" for i, prob in enumerate(pred[0] * 100)
    ]
    # Join and print the results
    output_string = ", ".join(formatted_predictions)
    print(output_string)

    gesture_counts = Counter(pred_labels)
    most_common_gesture = gesture_counts.most_common(1)[0][0]
    return most_common_gesture

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, pose, lh, rh]).tolist()

def draw_landmarks(image, model) -> None:
    mp_drawing.draw_landmarks(image, model.face_landmarks, mp_holistics.FACEMESH_CONTOURS,
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=1)
                            )
    mp_drawing.draw_landmarks(image, model.pose_landmarks, mp_holistics.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                            )
    mp_drawing.draw_landmarks(image, model.right_hand_landmarks, mp_holistics.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=mp_drawing.GREEN_COLOR, thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(181, 135, 230), thickness=2, circle_radius=2)
                            )
    mp_drawing.draw_landmarks(image, model.left_hand_landmarks, mp_holistics.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=mp_drawing.BLUE_COLOR, thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(181, 135, 230), thickness=2, circle_radius=2)
                            )


cap = cv2.VideoCapture(r"website\backend\saved_images\output_video_3ed5f6c8b66d448bb0b56ef8d97ff076.mp4")
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_number = 0 
csv_data = []
isRecording = True
while cap.isOpened():
    key = cv2.waitKey(5) & 0xFF
    match key:
        case ProgramShortcuts.quit.value:
            break

    ret, frame = cap.read()
    if not ret:
        break 

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistics.process(frame_rgb)
    draw_landmarks(frame, results)

    if isRecording:
        hol_landmarks = extract_keypoints(results)
        csv_data.append(hol_landmarks)

    
    cv2.imshow("Recording Gestures", frame)



# print(csv_data)
print(frame_rate)
print(frame_width)
print(frame_height)
if isRecording and len(csv_data) == 30:
    print("yes")
    print(predict(csv_data, frame_rate, frame_width, frame_height))
else:
    print("no")
cap.release()
cv2.destroyAllWindows()
