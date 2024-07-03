import csv
import os
import cv2 
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision 
import time


class Landmarks():
    def __init__(self):
        self.HandLandmarker = vision.HandLandmarker 
        self.HandLandmarkerOptions = vision.HandLandmarkerOptions 
        self.HandLandmarkerResult = None
        self.createLandmarker()

    def createLandmarker(self):
        def update_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.HandLandmarkerResult = result
            
        self.BaseOptions = python.BaseOptions
        
        landmark_options = self.HandLandmarkerOptions(
            base_options = self.BaseOptions(model_asset_path="model/hand_landmarker.task"),
            running_mode = vision.RunningMode.LIVE_STREAM,
            num_hands = 2, 
            min_hand_detection_confidence = 0.60,
            min_hand_presence_confidence= 0.60, 
            min_tracking_confidence = 0.70,
            result_callback = update_result
        )

        self.HandLandmarker = self.HandLandmarker.create_from_options(landmark_options)

    def detect_async(self, frame):
        try:
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
            self.HandLandmarker.detect_async(mp_image, timestamp_ms = int(time.time() * 1000))
            return self.HandLandmarkerResult

        except Exception as e:
            print("Error in detect_landmarks:", e)
            return None

def annotate_image(image, landmarker_result):
    if landmarker_result is None or not landmarker_result.hand_landmarks:
        return image

    hand_landmark_list = landmarker_result.hand_landmarks
    annotated_image = np.copy(image)
    hand_landmark_list_len = len(hand_landmark_list)
    for i in range(hand_landmark_list_len):
        hand_landmarks = hand_landmark_list[i]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )
    
        
    return annotated_image

def main():
    capture = cv2.VideoCapture(0)
    landmarker = Landmarks()
    
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    isRecording = False
    landmark_seq = []
    gesture_action = "THANK-YOU"

    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        
        landmarker_result = landmarker.detect_async(frame)
        if landmarker_result:
          
            if isRecording:
                landmarks = [lm for lm in landmarker_result.hand_landmarks]
                landmarks_flat = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
                landmark_seq.append(landmarks_flat)

            anno_frame = annotate_image(frame, landmarker_result)
        else:
            anno_frame = frame

        cv2.imshow("Recording Window",anno_frame)


        key = cv2.waitKey(5) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            if not isRecording:
                isRecording = True
                landmark_seq = []  # Reset the landmark sequence
                print("Recording started...")
        elif key == ord('s'):
            if isRecording:
                isRecording = False
                print("Recording stopped.")

                if landmark_seq:
                    cur_time = int(time.time_ns())
                    output_file = os.path.join(output_dir, f"{gesture_action}_{cur_time}.csv")

                    with open(output_file, 'w', newline='') as f:
                        writer = csv.writer(f)

                        # Write the header into the CSV
                        header = ['frame'] + [f'{coord}_{i}' for i in range(21) for coord in ('x', 'y', 'z')] + ['frame_rate', 'frame_width', 'frame_height']
                        writer.writerow(header)

                        # Write the data
                        for i, frame_data in enumerate(landmark_seq):
                            writer.writerow([i] + frame_data + [frame_rate, frame_width, frame_height])
        else:
            # No default action to break the loop
            pass
        
    capture.release()
    cv2.destroyAllWindows()


main()
"""





def get_annotated_image(frame, landmarker):
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
    timestamp = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp)

    return annotate_image(mp_image, RESULT)

"""