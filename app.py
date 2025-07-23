import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO('weights/yolov8n.pt')
PERSON_CLASS_ID = 0

st.set_page_config(page_title="YOLOv8 Real-time Human Detection", layout="wide")
st.title("YOLOv8 Real-Time Human Detection")
st.markdown("This app uses your webcam to detect people in real-time using YOLOv8 Nano (no OpenCV).")

# Helper to draw bounding boxes using PIL
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

# Streamlit WebRTC video processor
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = image[..., ::-1]  # Convert BGR to RGB for PIL

        start = time.time()
        result = self.model.predict(image_rgb)[0]
        image_out = draw_boxes(image_rgb, result.boxes)
        end = time.time()

        # Add FPS counter
        fps = 1 / (end - start) if (end - start) > 0 else 0
        image_pil = Image.fromarray(image_out)
        draw = ImageDraw.Draw(image_pil)
        draw.text((20, 30), f"FPS: {fps:.2f}", fill="red")
        return av.VideoFrame.from_ndarray(np.array(image_pil), format="rgb24")

# Start the webcam streamer
webrtc_streamer(
    key="yolo-stream",
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
