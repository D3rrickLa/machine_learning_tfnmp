import cv2
import mediapipe as mp
import os
import csv
import time
import threading

# Initialize MediaPipe Hands and Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the hand detector
hands = mp_hands.Hands()

# Initialize variables
isRecording = False
landmark_seq = []
landmark_world_seq = []
gesture_action = "ALL-DONE"
output_dir = "data"  # Ensure this directory exists

capture = cv2.VideoCapture(0)  # Initialize video capture, change source as needed

frame_rate = capture.get(cv2.CAP_PROP_FPS)
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

def countdown(seconds):
    for i in range(seconds, 0, -1):
        print(f"Recording starts in {i} seconds...")
        time.sleep(1)
    print("Recording started!")

def save_to_csv(gesture_action, landmark_seq, landmark_world_seq, frame_rate, frame_width, frame_height):
    cur_time = int(time.time_ns())
    output_file = os.path.join(output_dir, f"{gesture_action}_{cur_time}.csv")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write the header into the CSV
        header = (
            ['frame']
            + [f'{coord}_{i}' for i in range(21) for coord in ('x', 'y', 'z')]
            + [f'{coord}_{i}' for i in range(21) for coord in ('wx', 'wy', 'wz')]
            + ['frame_rate', 'frame_width', 'frame_height']
        )
        writer.writerow(header)

        # Write the data
        for i, (frame_data, wrld_frame_data) in enumerate(zip(landmark_seq, landmark_world_seq)):
            writer.writerow([i] + frame_data + wrld_frame_data + [frame_rate, frame_width, frame_height])

def auto_capture_gestures(repeats, countdown_seconds, capture_duration, gesture_action, frame_rate, frame_width, frame_height):
    global isRecording, landmark_seq, landmark_world_seq

    for _ in range(repeats):  # Repeat the recording process
        countdown(countdown_seconds)  # Perform the countdown
        isRecording = True
        landmark_seq = []  # Reset the landmark sequence
        landmark_world_seq = []  # Reset the world landmark sequence

        start_time = time.time()
        while time.time() - start_time < capture_duration:  # Record for the specified duration
            ret, frame = capture.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks, hand_world_landmarks in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    frame_data = []
                    wrld_frame_data = []
                    for lm, wlm in zip(hand_landmarks.landmark, hand_world_landmarks.landmark):
                        frame_data.extend([lm.x, lm.y, lm.z])
                        wrld_frame_data.extend([wlm.x, wlm.y, wlm.z])
                    if isRecording:
                        landmark_seq.append(frame_data)
                        landmark_world_seq.append(wrld_frame_data)

            cv2.imshow("Recording Window", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        isRecording = False
        print("Recording stopped.")
        if landmark_seq and landmark_world_seq:
            save_to_csv(gesture_action, landmark_seq, landmark_world_seq, frame_rate, frame_width, frame_height)

def start_auto_capture():
    auto_capture_gestures(
        repeats=5,
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

    # Convert the frame to RGB as MediaPipe uses RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_world_landmarks in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks):
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks and world landmarks and append to sequences if recording
            frame_data = []
            wrld_frame_data = []
            for lm, wlm in zip(hand_landmarks.landmark, hand_world_landmarks.landmark):
                frame_data.extend([lm.x, lm.y, lm.z])
                wrld_frame_data.extend([wlm.x, wlm.y, wlm.z])
            if isRecording:
                landmark_seq.append(frame_data)
                landmark_world_seq.append(wrld_frame_data)

    cv2.imshow("Recording Window", frame)

    key = cv2.waitKey(5) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('o'):
        # Start the auto capture process in a new thread
        threading.Thread(target=start_auto_capture).start()

# Release resources
capture.release()
cv2.destroyAllWindows()
