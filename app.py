import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO("weights/yolov8n.pt")
PERSON_CLASS_ID = 0

st.set_page_config(page_title="YOLOv8 Real-time Human Detection", layout="wide")
st.title("YOLOv8 Real-Time Human Detection")
st.markdown("This app uses your webcam to detect people in real-time using YOLOv8 Nano (no OpenCV, no av import).")

def draw_boxes(image_np, boxes):
    image_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)

    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if cls == PERSON_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, y1 - 10), f"Person: {conf:.2f}", fill="green")

    return np.array(image_pil)

class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = img[..., ::-1]

        start = time.time()
        result = self.model.predict(img_rgb, verbose=False)[0]
        processed = draw_boxes(img_rgb, result.boxes)

        fps = 1 / (time.time() - start + 1e-6)
        image_pil = Image.fromarray(processed)
        draw = ImageDraw.Draw(image_pil)
        draw.text((20, 30), f"FPS: {fps:.2f}", fill="red")

        return image_pil

# Launch the webcam
webrtc_streamer(
    key="yolo-stream",
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
