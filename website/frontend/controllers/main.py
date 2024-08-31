import json
from fastapi.staticfiles import StaticFiles
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
import websockets
from fastapi.templating import Jinja2Templates 
from fastapi.responses import FileResponse, HTMLResponse
import requests 
from starlette import websockets as sws
from PIL import Image as img
import zlib 

app = FastAPI(debug=True)
app.mount("/website/frontend/static", StaticFiles(directory=r"./website/frontend/static"), name="static")
templates = Jinja2Templates(directory=r"./website/frontend/templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)

# Route to serve the favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.websocket("/ws2")
async def websocket_point_2(websocket: WebSocket):
    await websocket.accept()
    buffer = bytearray()
    lt_buffer= bytearray()
    frames = []
    width, height = 640, 480 
    expected_size = width * height * 3 
    server_uri = "http://localhost:8001/process_video"
    try:
        while True: 
            data = await websocket.receive_bytes()   
            buffer.extend(data)
            lt_buffer.extend(data)

            if len(buffer) >= expected_size: 
                image = img.frombytes("RGB", (width, height), bytes(buffer[:expected_size]))
                frame = np.array(image)
                frames.append(frame)
                
                buffer = buffer[expected_size:] # removes the processed frame from the buffer?

                # NOTE: it can't be frames length because data might really be a single still image
                if len(lt_buffer) // expected_size == 30: 
                    print("data full, sending")
                    compressed_data = zlib.compress(json.dumps([f.tolist() for f in frames]).encode())
                    print(len(compressed_data))
                    response = requests.post(server_uri, data=compressed_data, headers={'Content-Encoding': 'gzip'})
                    print(response.text)
                    
                    frames = []
                    lt_buffer.clear()
    
    except WebSocketDisconnect:
        pass
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

# okay new ish problem, that vidoe being saved is like 2x faster. cruded way of fixing this is
# to on this end, hald the speed to 15 fps (need 30fps) and lie on the backend what fps we have
# should also consider batching the chunks, like 90 fps per chunk and sending them to the backend
# rather than 1 chunk to backend each time. We have the basis all here to send video data to the backend
# we would just need to do some csv style formatting before passing it to the model. look ath the demo from
# before in the models
# 
# 
# 
