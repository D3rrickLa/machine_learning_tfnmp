import os
import cv2
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect 
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates 
from fastapi.responses import FileResponse, StreamingResponse, PlainTextResponse
import numpy as np
import uvicorn 
import requests 
import asyncio
from starlette import websockets as sws

import uvicorn.config 

app = FastAPI(debug=True)
app.mount("/website/backend/static", StaticFiles(directory=r"./website/backend/static"), name="static")
save_dir = r"./website/backend/saved_images"

@app.get("/")
def index():
    return {"message" : "server is working properly"}

@app.websocket("/process_video")
async def process(websocket: WebSocket):
    await websocket.accept()

    try:
        # Receive data from the WebSocket connection
        data = await websocket.receive_bytes()
        print(f"Received data of length: {len(data)}")
        
        # Send a successful response back to the client
        response_message = "Successful"
        await websocket.send_text(response_message)


    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Backend Error: {e}")
    finally:
        if not websocket.client_state == sws.WebSocketState.DISCONNECTED:
            await websocket.close()

def main():
    config = uvicorn.Config("rest_controller:app", host="localhost", port=8001, reload=True)
    sever  = uvicorn.Server(config)
    sever.run()

if __name__ == "__main__":
    main()