import mediapipe as mp
import numpy as np 
import pandas as pd
import time
from typing import Counter

class Predictor():
    def __init__(self, preprocessor, model, class_labels: list) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.class_labels = class_labels
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistics = mp.solutions.holistic
        self.holistics = self.mp_holistics.Holistic(static_image_mode=False, min_detection_confidence=0.65, min_tracking_confidence=0.8)
        self.header = (
                [f'{coord}_{i}' for i in range(468) for coord in ('hx', 'hy', 'hz')]+
                [f'{coord}_{i}' for i in range(33) for coord in ('px', 'py', 'pz', "pose_visibility")]+
                [f'{coord}_{i}' for i in range(21) for coord in ('lx', 'ly', 'lz')]+
                [f'{coord}_{i}' for i in range(21) for coord in ('rx', 'ry', 'rz')]+
                ["frame_rate", "frame_width", "frame_height", "frame", "gesture_index"]
            )
        self.landmark_columns = [f"{col}" for col in self.header if col.startswith(("hx", "hy", "hz", "px", "py", "pz", "lx", "ly", "lz", "rx", "ry", "rz"))]
        self.categorical_columns =  ["gesture_index"]
        self.numerical_columns = ["frame", "frame_rate", "frame_width", "frame_height"] + [f"{col}" for col in self.header if col.startswith("pose_visibility")]

    def draw_landmarks(self, image, model) -> None:
        self.mp_drawing.draw_landmarks(image, model.face_landmarks, self.mp_holistics.FACEMESH_CONTOURS,
                                    self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=1),
                                    self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=1)
                                )
        self.mp_drawing.draw_landmarks(image, model.pose_landmarks, self.mp_holistics.POSE_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )
        self.mp_drawing.draw_landmarks(image, model.right_hand_landmarks, self.mp_holistics.HAND_CONNECTIONS, 
                                    self.mp_drawing.DrawingSpec(color=self.mp_drawing.GREEN_COLOR, thickness=2, circle_radius=4),
                                    self.mp_drawing.DrawingSpec(color=(181, 135, 230), thickness=2, circle_radius=2)
                                )
        self.mp_drawing.draw_landmarks(image, model.left_hand_landmarks, self.mp_holistics.HAND_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(color=self.mp_drawing.BLUE_COLOR, thickness=2, circle_radius=4),
                                    self.mp_drawing.DrawingSpec(color=(181, 135, 230), thickness=2, circle_radius=2)
                                )

    def extract_keypoints(self, results) -> list:

        # Process pose landmarks (if available)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([face, pose, lh, rh]).tolist()

    def predict(self, landmark_seq, frame_rate=30, frame_width=640, frame_height=480) -> str:
        
        data = [frame_data + [frame_rate, frame_width, frame_height, i, time.time_ns()]  for i, frame_data in enumerate(landmark_seq)]
        df = pd.DataFrame(data, columns=self.header)
        
        X_new_pre = self.preprocessor.transform(df[self.landmark_columns + self.numerical_columns + self.categorical_columns])
        X_new_pre.drop(columns="remainder__gesture_index", inplace=True)
        
        X_new = np.reshape(X_new_pre, (1, X_new_pre.shape[0], X_new_pre.shape[1]))
        pred = self.model.predict(X_new, verbose=0)

        pred_labels = [self.class_labels[np.argmax(p)] for p in pred]
        gesture_counts = Counter(pred_labels)
        most_common_gesture = gesture_counts.most_common(1)[0][0]

        return most_common_gesture
