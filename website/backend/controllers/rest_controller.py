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

@app.post("/process_video")
async def process(request: Request):
    try:
        data = await request.body()
        print(len(data))
        return ("got message")
    except Exception as e: 
        print(f"Error: {e}") 


def main():
    config = uvicorn.Config("rest_controller:app", host="localhost", port=8001, reload=True)
    sever  = uvicorn.Server(config)
    sever.run()

if __name__ == "__main__":
    main()