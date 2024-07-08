import csv
import os
import cv2 
import numpy as np
import mediapipe as mp
import time
import keyboard


def main():
    capture = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    mp_drawing = mp.solutions.drawing_utils
    
    output_dir = "data/data_2"
    os.makedirs(output_dir, exist_ok=True)

    isRecording = False
    landmark_seq = []
    landmark_world_seq = []
    hand_seq = []
    gesture_action = "THANK-YOU" # Change this 

    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        

        if results.multi_hand_landmarks and results.multi_hand_world_landmarks:         
            for hand_landmarks, hand_world_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lbl = [cls.label for cls in handedness.classification][0]
                if lbl == "Left":
                    cv2.putText(frame, lbl, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, lbl, (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

                if isRecording:
                    get_landmarks(landmark_seq, hand_landmarks)
                    get_landmarks(landmark_world_seq, hand_world_landmarks)       
                    get_handedness(hand_seq, handedness)

        cv2.imshow("Hand Gesture Recording", frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if not isRecording:
                isRecording = True 
                landmark_seq = []
                landmark_world_seq = [] 
                print("Recording started...")

        elif key == ord('s'):
            if isRecording:
                isRecording = False 
                print("Recording Stopped")

                if landmark_seq and landmark_world_seq:
                    cur_time = int(time.time_ns())
                    output_file = os.path.join(output_dir, f"{gesture_action}_{cur_time}.csv")

                    with open(output_file, 'w', newline='') as f:
                        writer = csv.writer(f)

                        # Write the header into the CSV
                        header = ['frame'] + [f'{coord}_{i}' for i in range(21) for coord in ('x', 'y', 'z')] + [f'{coord}_{i}' for i in range(21) for coord in ('wx', 'wy', 'wz')] + ["hand", "score"] + ['frame_rate', 'frame_width', 'frame_height']
                        writer.writerow(header)

                        # Write the data
                        for i, (frame_data, wrld_frame_data, hand_data) in enumerate(zip(landmark_seq, landmark_world_seq, hand_seq)):
                            writer.writerow([i] + frame_data + wrld_frame_data + hand_data + [frame_rate, frame_width, frame_height])
        
        elif key == ord('o'):
            n_seconds = 3
            n_times = 5
            how_long = 3
            for _ in range(n_times):
                cv2.imshow("Hand Gesture Recording", frame)
                for i in range(n_seconds, 0, -1):
                    print(f"Recording starts in {i} seconds")
                    time.sleep(1)
                
                print("Recording started")
                start_time = time.time()
                keyboard.press_and_release('r')
                while time.time() - start_time < how_long:
                    continue

                print("Recording Ended\n")
                keyboard.press_and_release('s')
        else:
            # No default action to break the loop
            pass

    capture.release()
    cv2.destroyAllWindows()

def get_landmarks(lm_seq, hand_lm):
    landmarks = [lm for lm in hand_lm.landmark]
    landmarks_flat = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
    lm_seq.append(landmarks_flat)


def get_handedness(hand_seq, hand_lm):
    handedness_flat = [] 
    for handedness in hand_lm.classification:
        handedness_flat.append(handedness.label)
        handedness_flat.append(handedness.score)

    hand_seq.append(handedness_flat)

main()