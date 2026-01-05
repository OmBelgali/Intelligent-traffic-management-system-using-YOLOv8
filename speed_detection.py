import cv2
import numpy as np
import tempfile
import gradio as gr
from ultralytics import YOLO
import torch

# ==================== MODEL ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("Weights/yolov8s.pt").to(device)

# ==================== CLASSES ====================
with open("coco.txt", "r") as f:
    class_list = f.read().splitlines()

# ==================== ROIs ====================
area1 = [(314, 297), (742, 297), (805, 323), (248, 323)]
area2 = [(171, 359), (890, 359), (1019, 422), (15, 422)]

# ==================== PARAMETERS ====================
SPEED_LIMIT = 120       # km/h
DISTANCE_M = 45         # meters between ROI1 and ROI2
OUT_W, OUT_H = 1020, 500

# ==================== MAIN FUNCTION ====================
def speed_detection(video_path):
    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)

    # -------- FPS FIX (CRITICAL) --------
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps == 0:
        fps = 25

    # -------- OUTPUT VIDEO --------
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = temp_out.name

    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (OUT_W, OUT_H)
    )

    # -------- TRACKING DATA --------
    enter_frame = {}
    exit_frame = {}
    vehicle_ids = set()
    frame_count = 0
    written_frames = 0

    # ==================== LOOP ====================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (OUT_W, OUT_H))

        # -------- YOLO + BYTE TRACK --------
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4,
            verbose=False
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy()

            for box, cls_id, tid in zip(boxes, class_ids, track_ids):
                tid = int(tid)
                class_name = class_list[int(cls_id)]

                if class_name not in ["car", "motorcycle", "bus", "truck"]:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                in_area1 = cv2.pointPolygonTest(
                    np.array(area1), (cx, cy), False
                ) >= 0
                in_area2 = cv2.pointPolygonTest(
                    np.array(area2), (cx, cy), False
                ) >= 0

                # -------- ENTRY / EXIT --------
                if in_area1 and tid not in enter_frame:
                    enter_frame[tid] = frame_count

                if in_area2 and tid in enter_frame and tid not in exit_frame:
                    exit_frame[tid] = frame_count

                # -------- SPEED --------
                if tid in enter_frame and tid in exit_frame:
                    elapsed_frames = exit_frame[tid] - enter_frame[tid]
                    elapsed_time = elapsed_frames / fps

                    if elapsed_time > 0:
                        speed = (DISTANCE_M / elapsed_time) * 3.6

                        cv2.putText(
                            frame,
                            f"{int(speed)} km/h",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )

                        if speed > SPEED_LIMIT:
                            cv2.putText(
                                frame,
                                "SPEED LIMIT VIOLATED",
                                (300, 80),
                                cv2.FONT_HERSHEY_TRIPLEX,
                                1,
                                (0, 0, 255),
                                2
                            )

                vehicle_ids.add(tid)

                # -------- DRAW BOX --------
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"ID:{tid}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

        # -------- DRAW ROIs --------
        cv2.polylines(frame, [np.array(area1)], True, (0, 255, 0), 2)
        cv2.polylines(frame, [np.array(area2)], True, (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Vehicle Count: {len(vehicle_ids)}",
            (380, 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (102, 0, 255),
            2
        )

        out.write(frame)
        written_frames += 1

    cap.release()
    out.release()

    print("Frames written:", written_frames)
    return out_path

# ==================== GRADIO UI ====================
interface = gr.Interface(
    fn=speed_detection,
    inputs=gr.Video(label="Upload Traffic Video"),
    outputs=gr.Video(label="Speed Evaluation Output"),
    title="YOLOv8 Vehicle Speed Detection System",
    description=(
        "Detect vehicles, track them using ByteTrack, and calculate speed "
        "using ROI-based distance measurement. "
        "Processing time depends on video length."
    )
)

interface.launch()
