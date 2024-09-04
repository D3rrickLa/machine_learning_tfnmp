import datetime
import io
import os
import subprocess
import cv2
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles 
from fastapi.templating import Jinja2Templates 
from fastapi.responses import FileResponse, StreamingResponse, PlainTextResponse, Response 
import numpy as np 
import uvicorn 
from starlette import websockets
import uuid
import mediapipe as mp

app = FastAPI(debug=True)

@app.get("/")
def index():
    return "the server is working properly"

@app.post("/process")
async def process(request: Request):
    try:
        output_dir = "website/backend/saved_images"
        os.makedirs(output_dir, exist_ok=True)
        
        unique_filename = os.path.join(output_dir, f"output_video_{uuid.uuid4().hex}.mp4")
        
        ffmpeg_command = [
            "ffmpeg",
            "-hide_banner",
            "-f", "rawvideo",                # Input format is raw video
            "-pix_fmt", "rgb24",             # Pixel format
            "-s", "640x480",                 # Resolution
            "-r", "30",                      # Frame rate
            "-i", "-",                       # Read input from stdin
            "-an",
            "-c:v", "libx264",               # Video codec
            "-preset", "ultrafast",          # Encoding speed
            "-tune", "zerolatency",
            "-crf", "23",                    # Quality factor
            "-b:v", "1M",                    # Video bitrate
            "-vf", "scale=640:480",          # Scale filter
            "-movflags", "frag_keyframe+empty_moov", # Fix typo
            "-f", "mp4",                     # Output format
            unique_filename                  # Output file
            # "udp://localhost:8888?pkt_size=1316"  # Output to UDP port with packet size
        ]
        output_buffer = process_video_frames(await request.body())
        # Run the ffmpeg command
        with subprocess.Popen(
            ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            stdout, stderr = process.communicate(input=output_buffer)
            if process.returncode != 0:
                raise Exception(f"ffmpeg error: {stderr.decode()}")

        return Response(content=f"Video saved as {unique_filename}", status_code=200)

    except Exception as e: 
        return print(f"Error occured during video processing: {e}")

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
def process_video_frames(input_data):
        # Set up MediaPipe Face Mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5
    )

    # Convert input data to video stream
    cap = cv2.VideoCapture(io.BytesIO(input_data))

    # Buffer to store the output video
    output_buffer = io.BytesIO()

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_buffer, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame for face mesh landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
                )

        # Write frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    face_mesh.close()

    # Return video as a bytes stream
    output_buffer.seek(0)
    return output_buffer
        


def main():
    config = uvicorn.Config("back_controller:app", host="localhost", port=8001, reload="true")
    uvicorn.Server(config=config).run()



if __name__ == "__main__":
    main()


""""
    'ffmpeg',
    '-re', 
    '-rtbufsize', '10M', 
    '-f', 'dshow', 
    '-i', 'video=Integrated Camera',
    '-an', 
    '-pix_fmt', 'yuv420p', 
    '-c:v', 'libx264', 
    '-preset', 'ultrafast',
    '-tune', 'zerolatency', 
    '-crf', '24', 
    '-b:v', '500k', 
    '-r', '10', 
    '-vf', 'scale=640:480',
    '-t', '5',
    '-f', 'h264', 
    'pipe:'


    'ffmpeg',
    '-threads', '0',
    '-nostats',
    '-loglevel', '-8',
    '-v', 'quiet',
    '-probesize', '8192',
    '-hide_banner',
    '-i', 'pipe:',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-an',
    '-sn',
    'pipe:'

    


"""