import cv2 
import os 

# Function to extract frames from a video
def extract_frames(video_folder, output_folder):
    for filename in os.listdir(video_folder):
        video_path = os.path.join(video_folder, filename)
        if os.path.isfile(video_path):
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame
                frame_path = os.path.join(output_folder, f"{filename}_frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)

                frame_count += 1

            cap.release()

# Example usage
video_folder = "data/dataset1/swipe_right"
output_folder = "data/dataset1_extracted/swipe_right"

extract_frames(video_folder, output_folder)