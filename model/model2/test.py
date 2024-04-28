import cv2 
import mediapipe as mp 
import numpy as np 
from mediapipe import solutions 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 
from mediapipe.framework.formats import landmark_pb2 
import time 


MODEL_PATH = 'model/model2/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (212, 130, 173)


class LandMarker():
    def __init__(self):
        self.result = HandLandmarkerResult
        self.landmarker = HandLandmarker
        self.createLandmarker()
    
    def createLandmarker(self):
        
        def update_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result



        options = HandLandmarkerOptions(
            base_options = BaseOptions(model_asset_path=MODEL_PATH),
            running_mode = VisionRunningMode.LIVE_STREAM,
            num_hands = 1, # 2 hands is not working
            min_hand_detection_confidence = 0.7,
            min_hand_presence_confidence = 0.7,
            min_tracking_confidence = 0.7,
            result_callback = update_result

        )

        self.landmarker = self.landmarker.create_from_options(options)

    def detect_async(self, frame):
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
        self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))
    
    def close(self):
        self.landmarker.close()

def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarker):
    try:
        if detection_result.hand_landmarks == []:
            return rgb_image
        else:
            hand_landmarks_list = detection_result.hand_landmarks
            annotated_image = rgb_image

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                
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


                return annotated_image
    except:
        return rgb_image
    
capture = cv2.VideoCapture(0)
hand_landmark = LandMarker()

while(True):
    ret, frame = capture.read()
    hand_landmark.detect_async(frame)
    frame = draw_landmarks_on_image(frame, hand_landmark.result)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break 

capture.release()
cv2.destroyAllWindows()