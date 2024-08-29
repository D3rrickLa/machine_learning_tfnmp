import os
import cv2
from fastapi import FastAPI, Request, WebSocket 
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates 
from fastapi.responses import FileResponse, StreamingResponse, PlainTextResponse
import numpy as np
import uvicorn 
import requests 
import asyncio

import uvicorn.config 

app = FastAPI()
app.mount("/static", StaticFiles(directory=r"./website/backend/static"), name="static")
save_dir = r"./website/backend/saved_images"
@app.get("/")
def index():
    return {"message" : "server is working properly"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.post("/process_video")
async def process_video(request: Request):
    data = await request.body()
    np_arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Failed to decode image frame."}

    # Define the file path where the frame will be saved
    frame_filename = os.path.join(save_dir, "frame.jpg")

    # Save the frame as a .jpg image
    cv2.imwrite(frame_filename, frame)
    return {"message" : "Video processed successfully"}




def main():
    config = uvicorn.Config("rest_controller:app", host="localhost", port=8001, reload=True)
    sever  = uvicorn.Server(config)
    sever.run()

if __name__ == "__main__":
    main()