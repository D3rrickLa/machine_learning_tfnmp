import os
import time
import unittest 
import cv2
from fastapi.testclient import TestClient
from website.backend.controllers.back_controller import app
client = TestClient(app)

class TestBackendProcess(unittest.TestCase):
    def initTest(self):
        response = client.get("/")
        self.assertEquals("the server is working properly", response.text)
    
    def testProcess(self):
        video_path = r"C:\Users\Gen3r\Documents\capstone\ml_model\test\saved_videos\output_video_3ed5f6c8b66d448bb0b56ef8d97ff076.mp4"
        
        if not os.path.exists(video_path):
            self.fail("VIdeo file not found")
        
        video_lists = []
        for i in range(100):
            video_lists.append(video_path)

            time_1 = time.time()
            cap = cv2.VideoCapture(video_lists[i])

            frame_bytes = []
            success, frame = cap.read()

            while success:
                frame_bytes.append(frame.tobytes())
                success, frame = cap.read() 
            
            cap.release() 

            body = b''.join(frame_bytes)

            response = client.post("/process", content=body)

            print(f"total time: {time.time() - time_1} | current: {i}")
        
        self.assertEqual(response.status_code, 200)  # Check if the response is OK
        self.assertIn("Video was decoded", response.text)  # Check for expected response content

if __name__ == "__main__":
    unittest.main()
        

