import asyncio
import os
import time 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response 
import joblib
import numpy as np 
import pandas as pd
import uvicorn 
from keras import models
from website.backend.services.decoder import Decoder
from website.backend.services.predictor import Predictor
from starlette import websockets


app = FastAPI(debug=True)

model = models.load_model(r"model\CNN_LSTM\v3\models\model_11_v3_1724733396947051100.keras")
class_labels = pd.read_csv(r"model\CNN_LSTM\v3\class_labels.csv")["gesture"].tolist()
preprocessor = joblib.load(r"model\CNN_LSTM\v3\preprocessing_pipeline.pkl")


@app.get("/")
def index():
    return "the server is working properly"

# NOTE, the await feature can be the problem
# like the data is too much
@app.post("/process")
async def process(request: Request):
    landmark_seq = []
    decoder = Decoder()
    predictor = Predictor(preprocessor, model, class_labels)
    try:  
        frame_bytes = decoder.decode(await request.body())
        for i in range(len(frame_bytes)):
            frame = np.frombuffer(frame_bytes[i], np.uint8).reshape((480, 640, 3)).copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = predictor.holistics.process(frame_rgb)
            landmark_seq.append(predictor.extract_keypoints(results))

            if len(landmark_seq) == 30:
                pred_gesture = predictor.predict(landmark_seq)
                print(pred_gesture)
     
        return Response(content=f"Video was decoded: {pred_gesture}", status_code=200)

    except Exception as e:
        print(e)
        return Response(content="An error occurred during video processing", status_code=500)

    finally:
        del decoder 
        del predictor
        cv2.destroyAllWindows()


# @app.websocket("/ws")
# async def weksocket_process(websocket: WebSocket):
#     await websocket.accept()
#     print("WebSocket connection accepted")
#     buff = bytearray()
#     landmark_seq = []
#     decoder = Decoder()
#     predictor = Predictor(preprocessor, model, class_labels)
#     pred_gesture = None
#     try:  
#         while True:
#             time_1 = time.time()
#             data = await websocket.receive_bytes()
#             buff.extend(data)  

#             if(len(buff) == 921600 * 30):
#                 frame_bytes = decoder.decode(buff)
                
#                 for i in range(len(frame_bytes)):
#                     frame = np.frombuffer(frame_bytes[i], np.uint8).reshape((480, 640, 3))
#                     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     results = predictor.holistics.process(frame_rgb)
#                     landmark_seq.append(predictor.extract_keypoints(results))

#                     if len(landmark_seq) == 30:
#                         pred_gesture = predictor.predict(landmark_seq)

#                 await websocket.send_text(f"Received your data: {pred_gesture}")
#                 print(f"{time.time() - time_1} | {pred_gesture}")
#                 buff.clear()
#                 landmark_seq.clear()
            
#     except WebSocketDisconnect:
#         print("Client disconnected")
#     except Exception as e:
#         print("Error:", str(e))

@app.websocket("/ws")
async def weksocket_process(websocket: WebSocket):
    await websocket.accept()
    videobuffer = bytearray()
    try:
        while True:
            data = await websocket.receive_bytes()
            videobuffer.extend(data)
            if (len(videobuffer) == 921600 * 30):
                await process_video(videobuffer, websocket)
                videobuffer.clear()
            else:
                await websocket.send_text("doing work")

    except WebSocketDisconnect:
        print("Client has disconnected.")
    except Exception as e:
        print(f"WS Error: {e}")    

async def process_video(frames: bytearray, websocket: WebSocket):
 
    decoder = Decoder() 
    predictor = Predictor(preprocessor, model, class_labels)
    landmark_seq = []
    pred_gesture = None

    decoded_frame_list = decoder.decode(frames)
    print(len(frames))
    for i in range(len(decoded_frame_list)):
        frame_rgb = cv2.cvtColor(np.frombuffer(decoded_frame_list[i], np.uint8).reshape((480, 640, 3)), cv2.COLOR_BGR2RGB)
 
        results = predictor.holistics.process(frame_rgb)
        landmark_seq.append(predictor.extract_keypoints(results))

    if len(landmark_seq) == 30:
        pred_gesture = predictor.predict(landmark_seq)
        print(pred_gesture)

    await websocket.send_text(f"result: {pred_gesture}")


def main():
    config = uvicorn.Config("back_controller:app", host="localhost", port=8001, reload="true")
    uvicorn.Server(config=config).run()



if __name__ == "__main__":
    main()

# fastapi dev website\backend\controllers\back_controller.py --port 8001