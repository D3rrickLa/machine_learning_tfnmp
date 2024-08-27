import numpy as np
import mediapipe as mp

class HolisticLandmarks:

    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistics = mp.solutions.holistic
        self.holistics = self.mp_holistics.Holistic(static_image_mode=False, min_detection_confidence=0.65, min_tracking_confidence=0.8)

    def extract_keypoints(results) -> list:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([face, pose, lh, rh]).tolist() 
    
    def draw_landmarks(self, image, model) -> None:
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