import asyncio
import gc
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
from concurrent.futures import ThreadPoolExecutor


FLAG = False
app = FastAPI(debug=True)
exe = ThreadPoolExecutor()
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
    time_1 = time.time()

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
                # print(pred_gesture)
        
        print(f"total time POST: {time.time() - time_1}")
        return Response(content=f"Video was decoded: {pred_gesture}", status_code=200)

    except Exception as e:
        print(e)
        return Response(content="An error occurred during video processing", status_code=500)

    finally:
        del decoder 
        del predictor
        cv2.destroyAllWindows()



@app.post("/stop_signal")
async def process(reqiest: Request):
    return 0

# NOTE there is still pausing on the JS side when running asyncio
@app.websocket("/ws")
async def weksocket_process(websocket: WebSocket):
    await websocket.accept()
    videobuffer = bytearray()
    decoder = Decoder() 
    predictor = Predictor(preprocessor, model, class_labels)
    try:
        while True:
            data = await websocket.receive_bytes()
            videobuffer.extend(data)
            if (len(videobuffer) == 921600 * 30):
                asyncio.create_task(process_async(videobuffer.copy(), websocket, decoder, predictor))  # Copy current buffer to avoid erase conditions
                videobuffer.clear()
                continue
            else:
                await websocket.send_text("doing work")

    except WebSocketDisconnect:
        print("Client has disconnected.")
    except Exception as e:
        print(f"WS Error: {e}")    
    finally: 
        gc.collect()     
        pass


async def process_async(buffer, websocket: WebSocket, decoder, predictor):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(exe, process_video, buffer, decoder, predictor)
    await websocket.send_text(f"Result: {result}")

def process_video(frames: bytearray, decoder: Decoder, predictor: Predictor):
    decoded_frame_list = decoder.decode(frames) 
    landmark_seq = [predictor.extract_keypoints(predictor.holistics.process(cv2.cvtColor(np.frombuffer(i, np.uint8).reshape((480, 640, 3)),cv2.COLOR_BGR2RGB))) for i in decoded_frame_list]

    if len(landmark_seq) == 30:
        pred_gesture = predictor.predict(landmark_seq)  
        return pred_gesture
            
    return None 

def main():
    config = uvicorn.Config("back_controller:app", host="localhost", port=8001, reload="true")
    uvicorn.Server(config=config).run()

if __name__ == "__main__":
    main()

# fastapi dev website\backend\controllers\back_controller.py --port 8001


"""
prediction function - 0.2 sec 
landmark_seq = 1.2-7 sec
"""