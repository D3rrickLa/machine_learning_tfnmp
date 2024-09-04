import asyncio
import httpx
import requests
import uvicorn

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates 
from starlette import websockets

app = FastAPI(debug=True)
app.mount("/website/frontend/static", StaticFiles(directory=r"./website/frontend/static"), name="static")
templates = Jinja2Templates(directory=r"./website/frontend/templates")
STOP_PROCESSING = False

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.websocket("/ws")
async def websocket_point_2(websocket: WebSocket):
    global STOP_PROCESSING 
    await websocket.accept()
    bufferarr = bytearray()

    try:
        while True:
            data = await websocket.receive_bytes()
            bufferarr.extend(data)    
     
            if STOP_PROCESSING:
                STOP_PROCESSING = False
                bufferarr.clear()
                break

            if len(bufferarr) >= 921600 * 30:
                requests.post(url="http://localhost:8001/process", data=bufferarr)
                bufferarr.clear()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS Error: {e}")
    finally:
        if websocket.client_state != websockets.WebSocketState.DISCONNECTED:
            await websocket.close()

@app.post("/stop_processing")
async def stop_processing():
    global STOP_PROCESSING 
    STOP_PROCESSING = True
    return JSONResponse(content={"message": "Processing stopped"}, status_code=200)


def main():
    config = uvicorn.Config("main_controller:app", host="localhost", port=8000, reload="true")
    uvicorn.Server(config=config).run()

if __name__ == "__main__":
    main()