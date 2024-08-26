import cv2
import uvicorn
from fastapi import FastAPI, Request, Response 
from fastapi.templating import Jinja2Templates 
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse 
import requests 
import asyncio 
import threading 





app = FastAPI()
templates = Jinja2Templates(directory=r"./website/frontend/templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)

@app.get("/video")
def getVideoFeed(request: Request):
    return templates.TemplateResponse(name="table.html", request=request)

def main():
    config = uvicorn.Config("main:app", host="localhost", port=8080, reload=True)
    sever  = uvicorn.Server(config)
    sever.run()

if __name__ == "__main__":
    main()