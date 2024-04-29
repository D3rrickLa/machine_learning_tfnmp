# will record video clips, 5 seconds of a gesture
# will have 3 inital gestures, swipe left/right, and waving


import cv2
import time
FOLDER_PATH = "data/dataset1/"
GESTURE_FOLDER = "swipe_left" # change this

current_time = int(time.time())

camera = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'{FOLDER_PATH}{GESTURE_FOLDER}/output_{current_time}.avi', fourcc, 30.0, (640, 480))

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    cv2.imshow("frame", frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
camera.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()