import datetime
import io
import os
import subprocess
import time
from typing import Counter
import cv2
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles 
from fastapi.templating import Jinja2Templates 
from fastapi.responses import FileResponse, StreamingResponse, PlainTextResponse, Response 
import joblib
import numpy as np 
import pandas as pd
import uvicorn 
from starlette import websockets
import uuid
import mediapipe as mp
from keras import models

app = FastAPI(debug=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistics = mp.solutions.holistic
holistics = mp_holistics.Holistic(static_image_mode=False, min_detection_confidence=0.65, min_tracking_confidence=0.8)

model = models.load_model(r"model\CNN_LSTM\v3\models\model_11_v3_1724733396947051100.keras")
class_labels = pd.read_csv(r"model\CNN_LSTM\v3\class_labels.csv")["gesture"].tolist()
preprocessor = joblib.load(r"model\CNN_LSTM\v3\preprocessing_pipeline.pkl")

def draw_landmarks(image, model) -> None:
    mp_drawing.draw_landmarks(image, model.face_landmarks, mp_holistics.FACEMESH_CONTOURS,
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=1)
                            )
    mp_drawing.draw_landmarks(image, model.pose_landmarks, mp_holistics.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                            )
    mp_drawing.draw_landmarks(image, model.right_hand_landmarks, mp_holistics.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=mp_drawing.GREEN_COLOR, thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(181, 135, 230), thickness=2, circle_radius=2)
                            )
    mp_drawing.draw_landmarks(image, model.left_hand_landmarks, mp_holistics.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=mp_drawing.BLUE_COLOR, thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(181, 135, 230), thickness=2, circle_radius=2)
                            )

def extract_keypoints(results):

    # Process pose landmarks (if available)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, pose, lh, rh]).tolist()

def predict(landmark_seq, frame_rate, frame_width, frame_height):
    header = (
            [f'{coord}_{i}' for i in range(468) for coord in ('hx', 'hy', 'hz')]+
            [f'{coord}_{i}' for i in range(33) for coord in ('px', 'py', 'pz', "pose_visibility")]+
            [f'{coord}_{i}' for i in range(21) for coord in ('lx', 'ly', 'lz')]+
            [f'{coord}_{i}' for i in range(21) for coord in ('rx', 'ry', 'rz')]+
            ["frame_rate", "frame_width", "frame_height", "frame", "gesture_index"]
        )
    
    data = [
        frame_data + [frame_rate, frame_width, frame_height, i, time.time_ns()]  for i, frame_data in enumerate(landmark_seq)
    ]

    df = pd.DataFrame(data, columns=header)
    
    landmark_columns = [f"{col}" for col in df.columns if col.startswith(("hx", "hy", "hz", "px", "py", "pz", "lx", "ly", "lz", "rx", "ry", "rz"))]
    categorical_columns = ["gesture_index"]
    numerical_columns = ["frame", "frame_rate", "frame_width", "frame_height"] + [f"{col}" for col in df.columns if col.startswith("pose_visibility")]

    X_new_pre = preprocessor.transform(df[landmark_columns + numerical_columns + categorical_columns])
    X_new_pre.drop(columns="remainder__gesture_index", inplace=True)
    
    X_new = np.reshape(X_new_pre, (1, X_new_pre.shape[0], X_new_pre.shape[1]))
    pred = model.predict(X_new, verbose=0)

    pred_labels = [class_labels[np.argmax(p)] for p in pred]
    formatted_predictions = [
        f"{class_labels[i]}: {prob:.2f}" for i, prob in enumerate(pred[0] * 100)
    ]
    # Join and print the results
    output_string = ", ".join(formatted_predictions)
    print(output_string)

    gesture_counts = Counter(pred_labels)
    most_common_gesture = gesture_counts.most_common(1)[0][0]

    return most_common_gesture


@app.get("/")
def index():
    return "the server is working properly"

@app.post("/process")
async def process(request: Request):
    try:
        
        ffmpeg_command = [
            "ffmpeg",
            "-hide_banner",
            "-f", "rawvideo",                # Input format is raw video
            "-pix_fmt", "bgr24",             # Pixel format
            "-s", "640x480",                 # Resolution
            "-r", "30",                      # Frame rate
            "-i", "-",                       # Read input from stdin
            "-an",
            # "-c:v", "libx264",               # Video codec
            # "-preset", "ultrafast",          # Encoding speed
            # "-tune", "zerolatency",
            # "-crf", "23",                    # Quality factor
            "-b:v", "1M",                    # Video bitrate
            "-vf", "scale=640:480",          # Scale filter
            "-movflags", "frag_keyframe+empty_moov", # Fix typo
            "-f", "rawvideo",                     # Output format
            "-"
        ]
   
        # Run the ffmpeg command
        with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            process.stdin.write(await request.body())
            process.stdin.close()  # Close stdin after sending the data
            landmark_seq = []
            frame_width = 640
            frame_height = 480
            frame_rate = 30
            frame_size = frame_width * frame_height * 3  # 3 bytes per pixel (RGB)

            while True:
                frame_bytes = process.stdout.read(frame_size)
                if len(frame_bytes) < frame_size:
                    break 

                frame = np.frombuffer(frame_bytes, np.uint8).reshape((frame_height, frame_width, 3))
                frame_c = frame.copy()

                frame_rgb = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)
                results = holistics.process(frame_rgb)
                draw_landmarks(frame_rgb, results)
                cv2.imshow('Frame', frame_rgb)

                landmark_seq.append(extract_keypoints(results))

                if len(landmark_seq) == 30:
                    pred_gesture = predict(landmark_seq, frame_rate, frame_width, frame_height)
                    print(pred_gesture)
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        return Response(content=f"Video saved as", status_code=200)

    except Exception as e: 
        return print(f"Error occured during video processing: {e}")

    finally:
        cv2.destroyAllWindows()


def main():
    config = uvicorn.Config("back_controller:app", host="localhost", port=8001, reload="true")
    uvicorn.Server(config=config).run()



if __name__ == "__main__":
    main()

# fastapi dev website\backend\controllers\back_controller.py --port 8001