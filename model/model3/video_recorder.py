# will record video clips, 5 seconds of a gesture
# will have 3 inital gestures, swipe left/right, and waving

import cv2
import time

# Initialize video capture object
camera = cv2.VideoCapture(0)  # 0 indicates the default camera, you can specify a different camera index if needed

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Specify the codec (XVID) - other codecs are available, such as MJPG
FOLDER_PATH = "data/dataset1/"
GESTURE_FOLDER = "swipe_right" # change this
output_filename_prefix = "gesture_swipe_left_"  # Prefix for output file names
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = None
start_time = time.time()
record_duration = 3  # Duration to record in seconds
max_videos = 5  # Maximum number of videos to record

video_count = 0

while True:
    ret, frame = camera.read()
    if not ret:
        break

    cv2.imshow("frame", frame)

    # Check if it's time to start a new video
    if out is None or (time.time() - start_time) >= record_duration:
        if out is not None:
            out.release()  # Release the previous video writer if exists
            video_count += 1

        if video_count >= max_videos:
            print("Finished recording")
            break

        # Create a new output video file
        current_time = int(time.time())
        output_filename = f"{FOLDER_PATH}{GESTURE_FOLDER}/{output_filename_prefix}{video_count}_{current_time}.avi"
        out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))
        print(f"Recording to {output_filename}")

        start_time = time.time()  # Reset start time


    # Write the frame to the output video file
    out.write(frame)


    # Check for 'q' key press to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
camera.release()
if out is not None:
    out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()