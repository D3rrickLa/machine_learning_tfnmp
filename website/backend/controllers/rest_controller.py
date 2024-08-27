from fastapi import FastAPI, Request, WebSocket 
from fastapi.templating import Jinja2Templates 
from fastapi.responses import StreamingResponse, PlainTextResponse
import uvicorn 
import requests 
import asyncio

import uvicorn.config 

app = FastAPI()


@app.get("/")
def index():
    return {"this is working": True}



def main():
    config = uvicorn.Config("rest_controller:app", host="localhost", port=8000, reload=True)
    sever  = uvicorn.Server(config)
    sever.run()

if __name__ == "__main__":
    main()