import cv2
from fastapi.staticfiles import StaticFiles
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect 
from fastapi.templating import Jinja2Templates 
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, PlainTextResponse 
import requests 
import asyncio 
import threading 





app = FastAPI(debug=True)
app.mount("/static", StaticFiles(directory=r"./website/frontend/static"), name="static")
templates = Jinja2Templates(directory=r"./website/frontend/templates")


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
            print(len(data))
            response = requests.post("http://localhost:8001/process_video", data=data)

           
            await websocket.send_text(response.text) # Awaiting WebSocket to send the response

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        await websocket.close()


def main():
    config = uvicorn.Config("main:app", host="localhost", port=8080, reload=True)
    sever  = uvicorn.Server(config)
    sever.run()

if __name__ == "__main__":
    main()



# TODO
# figure out the ending of data frame frontend to backend - there's a reason why we used ffmpeg... 