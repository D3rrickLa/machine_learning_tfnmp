# this will use the live streaming feature of mediapipe, and not the frame -> image
# test of old code, because the new code didn't work with 2 hands
import cv2 
import numpy as np 
import mediapipe as mp
from mediapipe import solutions 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 
from mediapipe.framework.formats import landmark_pb2 
import time 
import subprocess

BaseOptions = python.BaseOptions 
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
GestureRecognizerResults = vision.GestureRecognizerResult
VisionRunningMode = vision.RunningMode


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (212, 130, 173)
# PATH = 'mediapipe_ml_builder/model/rps_recognizer.task'
PATH = 'model/model2/gesture_recognizer.task'


# code from https://medium.com/@oetalmage16/a-tutorial-on-finger-counting-in-real-time-video-in-python-with-opencv-and-mediapipe-114a988df46a
# modified by Derrick
class landmark_and_result():
    def __init__(self):
        self.result = GestureRecognizerResults
        self.gesture = GestureRecognizer 
        self.createLandmarker()
    
    def createLandmarker(self):
        def update_result(result: GestureRecognizerResults, output_image: mp.Image, timestamp_ms: int):
            self.result = result

        options = GestureRecognizerOptions(
            base_options = BaseOptions(model_asset_path=PATH),
            running_mode = VisionRunningMode.LIVE_STREAM,
            num_hands = 2,
            min_hand_detection_confidence = 0.7,
            min_hand_presence_confidence = 0.7,
            min_tracking_confidence = 0.7,
            result_callback = update_result
        )

        self.gesture = self.gesture.create_from_options(options)

    def detect_async(self, frame):
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
        self.gesture.recognize_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

    def close(self):
        self.landmarker.close()

def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarker):
    try:
      if detection_result.hand_landmarks == []:
         return rgb_image
      else:
         hand_landmarks_list = detection_result.hand_landmarks
         gestures_list = detection_result.gestures
         handedness_list = detection_result.handedness
         annotated_image = np.copy(rgb_image)

         # Loop through the detected hands to visualize.
         for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]
            gesture = gestures_list[idx]
            
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
               landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            mp.solutions.drawing_utils.draw_landmarks(
               annotated_image,
               hand_landmarks_proto,
               mp.solutions.hands.HAND_CONNECTIONS,
               mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
               mp.solutions.drawing_styles.get_default_hand_connections_style())
            
             # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # get top right corner of detected hand's bounding box
            text_x2 = int(max(x_coordinates) * width)
            text_y2 = int(max(y_coordinates) * height) - MARGIN

            # fix for problem of flipping the image and the labels also flip
            hand_label = None 
            if(handedness[0].category_name == "Left"):
                hand_label = "Right"
            elif(handedness[0].category_name == "Right"):
                hand_label = "Left"

            cv2.putText(annotated_image, f"{hand_label}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        
            cv2.putText(annotated_image, f"{gesture[0].category_name}",
                (text_x2, text_y2), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

         return annotated_image
    except:
      return rgb_image

hand_landmarker = landmark_and_result()
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    hand_landmarker.detect_async(frame)
    frame = draw_landmarks_on_image(frame,hand_landmarker.result)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break 

hand_landmarker.close()
camera.release()
cv2.destroyAllWindows()