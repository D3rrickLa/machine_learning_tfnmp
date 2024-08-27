import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect 
from fastapi.templating import Jinja2Templates 
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse 
import requests 
import asyncio 
import threading 





app = FastAPI(debug=True)
templates = Jinja2Templates(directory=r"./website/frontend/templates")

camera = cv2.VideoCapture(0)
camera_active = threading.Event()
camera_active.set()

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)

@app.get("/test")
def getTable(request: Request):
    return templates.TemplateResponse(name="table.html", request=request)

@app.websocket("/ws")
async def websocket_point(websocket: WebSocket):
    await websocket.accept()
    try:
        while True: 
            data = await websocket.receive_bytes()

            # convert byte stream to an image 
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


            # do something with the code, like result = process_frame(frame)

    except WebSocketDisconnect:
        print("Client disconnected")


def main():
    config = uvicorn.Config("main:app", host="localhost", port=8080, reload=True)
    sever  = uvicorn.Server(config)
    sever.run()

if __name__ == "__main__":
    main()