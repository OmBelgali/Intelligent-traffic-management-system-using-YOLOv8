import cv2
import math
import cvzone
import gradio as gr
from ultralytics import YOLO
import tempfile

# ---------------- LOAD MODELS ----------------
helmet_model = YOLO("Weights/best.pt")
plate_model = YOLO("Weights/number_plate.pt")

helmet_classes = ['With Helmet', 'Without Helmet']

# ---------------- MAIN FUNCTION ----------------
def helmet_detection_video(video):
    if video is None:
        return None

    # ✅ video is already a file path
    video_path = video

    cap = cv2.VideoCapture(video_path)

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = temp_out.name

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        helmet_results = helmet_model(img, conf=0.4, verbose=False)
        plate_results = plate_model(img, conf=0.4, verbose=False)

        for r in helmet_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = round(float(box.conf[0]), 2)

                label = helmet_classes[cls]
                color = (0, 255, 0) if label == "With Helmet" else (0, 0, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img,
                    f"{label} {conf}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        for r in plate_results:
            for box in r.boxes:
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                pconf = round(float(box.conf[0]), 2)

                cv2.rectangle(img, (px1, py1), (px2, py2), (255, 255, 0), 2)
                cv2.putText(
                    img,
                    f"Plate {pconf}",
                    (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

        out.write(img)

    cap.release()
    out.release()

    return out_path

# ---------------- GRADIO INTERFACE ----------------
interface = gr.Interface(
    fn=helmet_detection_video,
    inputs=gr.Video(label="Upload Traffic Video"),
    outputs=gr.Video(label="Helmet + Number Plate Detection Output"),
    title="YOLO Helmet & Number Plate Detection System",
    description="Detect helmet usage and vehicle number plates from traffic videos"
)

interface.launch()
