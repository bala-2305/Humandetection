import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from ultralytics import YOLO
import time

model = YOLO('weights/yolov8n.pt')
PERSON_CLASS_ID = 0  

st.set_page_config(page_title="YOLOv8 Real-time Human Detection", layout="wide")
st.title("YOLOv8 Real-Time Human Detection")

st.markdown("""
This app uses your webcam to detect people in real-time using YOLOv8 Nano.
""")

class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        start = time.time()
        result = self.model.predict(img)[0]

        for box in result.boxes:
            class_id = int(box.cls)
            conf = box.conf.item()
            if class_id == PERSON_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'Person: {conf:.2f}'
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        end = time.time()
        fps = 1 / (end - start) if (end - start) > 0 else 0
        cv2.putText(img, f'FPS: {fps:.2f}', (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="yolo-stream",
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
