import cv2 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision 
import time
import os 

BaseOptions = python.BaseOptions()
HandLandmarker = vision.HandLandmarker 
HandLandmarkerOptions = vision.HandLandmarkerOptions 
HandLandmarkerResult = vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode


def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))


def create_landmarker():

    landmark_options = HandLandmarkerOptions(
        base_options = BaseOptions(model_asset_path="model/hand_landmarker.task"),
        running_mode = VisionRunningMode.LIVESTREAM,
        num_hands = 2, 
        min_hand_presence_confidence=0.80, 
        min_tracking_confidence=0.80,
        result_callback=print_result
    )

    landmarker = HandLandmarker.create_from_options(landmark_options)

    return landmarker


def main():
    capture = cv2.VideoCapture(0)
    landmarker = create_landmarker()

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break 
        
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
        landmarker.detect_async(mp_image, int(time.time()))
