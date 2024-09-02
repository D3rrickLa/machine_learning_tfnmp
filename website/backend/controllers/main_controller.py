import io
import os
import subprocess
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles 
from fastapi.templating import Jinja2Templates 
from fastapi.responses import FileResponse, StreamingResponse, PlainTextResponse 
import numpy as np 
import uvicorn 
from starlette import websockets


app = FastAPI(debug=True)

@app.get("/")
def index():
    return "the server is working properly"

@app.post("/process")
async def process(request: Request):
    v_data = await request.body()

    ffmpeg_decode_cmd = [
        'ffmpeg', 
        '-f', 'mpegts', 
        '-i', 'pipe:0',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        'pipe:1'
    ]

    decode_process = subprocess.Popen(
        ffmpeg_decode_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Feed the incoming video data into the decode process
    decode_process.stdin.write(v_data)
    decode_process.stdin.close()

    while True: 
        raw_frame = decode_process.stdout.read(640 * 480 * 3)  # Adjust size for your resolution
        if not raw_frame:
            break
        frame = np.frombuffer(raw_frame, np.uint8).reshape((480, 640, 3))  # Adjust shape for your resolution
        print(len(frame))


def main():
    config = uvicorn.Config("main_controller:app", host="localhost", port=8001, reload="true")
    uvicorn.Server(config=config).run()



if __name__ == "__main__":
    main()