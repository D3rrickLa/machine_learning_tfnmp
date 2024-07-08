import csv
import os
import cv2 
import numpy as np
import mediapipe as mp
import time
import threading

# Initialize MediaPipe Hands and Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the hand detector
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Initialize variables
isRecording = False
landmark_seq = []
landmark_world_seq = []
hand_seq = []
gesture_action = "THANK-YOU"
output_dir = "data/data_2"  # Ensure this directory exists

capture = cv2.VideoCapture(0)  # Initialize video capture, change source as needed

frame_rate = capture.get(cv2.CAP_PROP_FPS)
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

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

def countdown(seconds):
    for i in range(seconds, 0, -1):
        print(f"Recording starts in {i} seconds...")
        time.sleep(1)
    print("Recording started!")

def save_to_csv(gesture_action, landmark_seq, landmark_world_seq, hand_seq, frame_rate, frame_width, frame_height):
    cur_time = int(time.time_ns())
    output_file = os.path.join(output_dir, f"{gesture_action}_{cur_time}.csv")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write the header into the CSV
        header = (
            ['frame']
            + [f'{coord}_{i}' for i in range(21) for coord in ('x', 'y', 'z')]
            + [f'{coord}_{i}' for i in range(21) for coord in ('wx', 'wy', 'wz')]
            + ['hand', 'score']
            + ['frame_rate', 'frame_width', 'frame_height']
        )
        writer.writerow(header)

        # Write the data
        for i, (frame_data, wrld_frame_data, hand_data) in enumerate(zip(landmark_seq, landmark_world_seq, hand_seq)):
            writer.writerow([i] + frame_data + wrld_frame_data + hand_data + [frame_rate, frame_width, frame_height])

def auto_capture_gestures(repeats, countdown_seconds, capture_duration, gesture_action, frame_rate, frame_width, frame_height):
    global isRecording, landmark_seq, landmark_world_seq, hand_seq

    for _ in range(repeats):  # Repeat the recording process
        countdown(countdown_seconds)  # Perform the countdown
        isRecording = True
        landmark_seq = []  # Reset the landmark sequence
        landmark_world_seq = []  # Reset the world landmark sequence
        hand_seq = []  # Reset the hand sequence

        start_time = time.time()
        while time.time() - start_time < capture_duration:  # Record for the specified duration
            ret, frame = capture.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
                for hand_landmarks, hand_world_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks, results.multi_handedness):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    lbl = [cls.label for cls in handedness.classification][0]
                    if lbl == "Left":
                        cv2.putText(frame, lbl, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, lbl, (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

                    frame_data = []
                    wrld_frame_data = []
                    for lm, wlm in zip(hand_landmarks.landmark, hand_world_landmarks.landmark):
                        frame_data.extend([lm.x, lm.y, lm.z])
                        wrld_frame_data.extend([wlm.x, wlm.y, wlm.z])
                    if isRecording:
                        landmark_seq.append(frame_data)
                        landmark_world_seq.append(wrld_frame_data)
                        get_handedness(hand_seq, handedness)

            cv2.imshow("Recording Window", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        isRecording = False
        print("Recording stopped.")
        if landmark_seq and landmark_world_seq and hand_seq:
            save_to_csv(gesture_action, landmark_seq, landmark_world_seq, hand_seq, frame_rate, frame_width, frame_height)

def start_auto_capture():
    auto_capture_gestures(
        repeats=20,
        countdown_seconds=3,
        capture_duration=3,
        gesture_action=gesture_action,
        frame_rate=frame_rate,
        frame_width=frame_width,
        frame_height=frame_height
    )

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    # Convert the frame to RGB as MediaPipe uses RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
        for hand_landmarks, hand_world_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks, results.multi_handedness):
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lbl = [cls.label for cls in handedness.classification][0]
            if lbl == "Left":
                cv2.putText(frame, lbl, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, lbl, (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

            # Extract landmarks and world landmarks and append to sequences if recording
            frame_data = []
            wrld_frame_data = []
            for lm, wlm in zip(hand_landmarks.landmark, hand_world_landmarks.landmark):
                frame_data.extend([lm.x, lm.y, lm.z])
                wrld_frame_data.extend([wlm.x, wlm.y, wlm.z])
            if isRecording:
                landmark_seq.append(frame_data)
                landmark_world_seq.append(wrld_frame_data)
                get_handedness(hand_seq, handedness)

    cv2.imshow("Recording Window", frame)

    key = cv2.waitKey(5) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        # Start the auto capture process in a new thread
        threading.Thread(target=start_auto_capture).start()

# Release resources
capture.release()
cv2.destroyAllWindows()


