import io
import os
import cv2
from fastapi.staticfiles import StaticFiles
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates 
from fastapi.responses import FileResponse, HTMLResponse
import requests 
import threading 
from starlette import websockets as sws
from PIL import Image as img

app = FastAPI(debug=True)
app.mount("/website/frontend/static", StaticFiles(directory=r"./website/frontend/static"), name="static")
templates = Jinja2Templates(directory=r"./website/frontend/templates")
save_dir = r'website/backend/saved_images'


camera = cv2.VideoCapture(0)
camera_active = threading.Event()
camera_active.set()

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)

# Route to serve the favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/test")
def getTable(request: Request):
    return templates.TemplateResponse(name="table.html", request=request)

@app.websocket("/ws")
async def websocket_point(websocket: WebSocket):
    await websocket.accept()
    try:
        while True: 
            data = await websocket.receive_bytes()
            response = requests.post("http://localhost:8001/process_video", data=data)

            await websocket.send_text(response.text) # Awaiting WebSocket to send the response

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        await websocket.close()

@app.websocket("/ws2")
async def websocket_point_2(websocket: WebSocket):
    await websocket.accept()
    frame_counter = 0 
    buffer = bytearray()
    try:
        while True: 
            data = await websocket.receive_bytes()   
            buffer.extend(data)

            width, height = 640, 480 

            expected_size = width * height * 3 

            if len(buffer) >= expected_size:
                image_stream = io.BytesIO(buffer)

                image = img.frombytes('RGB', (width, height), bytes(buffer[:expected_size]))

                frame_counter += 1 
                jpeg_filename = os.path.join(save_dir, f"frame_{frame_counter:04d}.jpeg")
                image.save(jpeg_filename)
                # Clear buffer after successful write
                buffer = buffer[expected_size:]


            
            else:
                print("waiting for more data")

    
    except WebSocketDisconnect:
        print("Websocket was disconnected.")
    except Exception as e:
        print(f"WS2 Error: {e}")

    finally:
        if not websocket.client_state == sws.WebSocketState.DISCONNECTED:
            await websocket.close()

def main():
    config = uvicorn.Config("main:app", host="localhost", port=8080, reload=True)
    sever  = uvicorn.Server(config)
    sever.run()

if __name__ == "__main__":
    main()



# TODO
# figure out the ending of data frame frontend to backend - there's a reason why we used ffmpeg... 
# OKAY SO ^^ is now cleared for now, we got images being sent to here (main.py)
# we can easily make this to acutal byte data and send it away for processing