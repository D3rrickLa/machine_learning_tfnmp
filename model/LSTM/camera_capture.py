import cv2 
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision 
import time


BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker 
HandLandmarkerOptions = vision.HandLandmarkerOptions 
HandLandmarkerResult = vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode

RESULT = 0
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global RESULT
    RESULT = result
    # print('hand landmarker result: {}'.format(result))


def create_landmarker():

    landmark_options = HandLandmarkerOptions(
        base_options = BaseOptions(model_asset_path="model/hand_landmarker.task"),
        running_mode = VisionRunningMode.LIVE_STREAM,
        num_hands = 2, 
        min_hand_presence_confidence=0.80, 
        min_tracking_confidence=0.80,
        result_callback=print_result
    )

    landmarker = HandLandmarker.create_from_options(landmark_options)

    return landmarker


def annotate_image(image, landmarker):
    """
    displays the landmarks onto the hands 
    """
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

    hand_landmark_list = landmarker.hand_landmarks 
    handedness_list = landmarker.handedness
    annotated_image = np.copy(image)

    for i in range(len(hand_landmark_list)):
        hand_landmarks = hand_landmark_list[i]
        handedness = handedness_list[i]

        # draw the hand landmarks 
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()        
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

        return annotated_image


def get_annotated_image(frame, landmarker):
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
    timestamp = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp)

    return annotate_image(mp_image, RESULT)


def main():
    capture = cv2.VideoCapture(0)
    landmarker = create_landmarker()

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break 
        
        
        anno_frame = get_annotated_image(frame, landmarker)
        cv2.imshow("Recording Window",anno_frame)


        key = cv2.waitKey(5) & 0xFF 
        if key == ord('q'):
            break
        
    capture.release()
    cv2.destroyAllWindows()


main()
