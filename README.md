# 🚦 Intelligent Traffic Monitoring System (YOLOv8)

## 📌 Overview
An AI-powered traffic surveillance system that analyzes road traffic videos to automatically detect **vehicles**, **helmet violations**, **overspeeding**, and **vehicle number plates**.  
The system combines deep learning–based object detection with multi-object tracking and speed estimation to identify traffic rule violations.

---

## 🚀 Key Highlights
- 🚗 **Vehicle Detection** using YOLOv8  
- ⛑ **Helmet Compliance Detection** (With / Without Helmet)  
- 🚓 **Overspeeding Detection** using ROI-based distance and time calculation  
- 🔢 **Number Plate Detection** (bounding box localization)  
- 🧠 **Multi-Object Tracking** with ByteTrack for consistent vehicle IDs  
- ⚡ **GPU-Accelerated Inference** using CUDA  
- 🌐 **Web Deployment** via Gradio for video upload and analysis  

---

## 📊 Performance
- **Detection Accuracy (mAP): 86%**  
- Optimized for faster inference using GPU acceleration

---

## 🛠 Tech Stack
- **Programming Language:** Python  
- **Computer Vision:** OpenCV  
- **Deep Learning:** YOLOv8 (Ultralytics)  
- **Tracking:** ByteTrack  
- **Deployment:** Gradio  
- **Hardware Acceleration:** NVIDIA CUDA  

---

## ▶ How to Run

### Install CUDA-enabled PyTorch (If GPU is Available)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run Helmet & Number Plate Detection
```bash
python helmetandnumberplate.py
```
### Run Speed Detection
```bash
python speed_detection.py
```
## 📌 Use Cases
- Smart traffic surveillance  
- Automated traffic rule enforcement  
- AI-based monitoring for smart city applications  

---

## 🔮 Future Scope
- OCR for number plate text recognition  
- Real-time camera feed integration  
- Database-backed violation logging  
- Deployment as a Flask-based REST API  

---

## 🎯 Why This Project Matters
This project demonstrates a **real-world application of computer vision and deep learning**, combining **object detection, multi-object tracking, speed estimation, and deployment** into a complete end-to-end intelligent traffic monitoring solution.


## 🗂️ Project Structure

```plaintext
Traffic_Violation_Project/
│
├── Media/
│   ├── 3524792219-preview.mp4
│   ├── sample_video_1.webm
│   └── vid1.mp4
│
├── Output/
│   ├── Gradio Interface.jpeg
│   ├── Helmet and Number Plate Detection Output.jpeg
|	├── Helmet and Number Plate Detection Output 1.jpeg
│   └── Overspeeding Violation Detection Output.jpeg
│
├── Weights/
│   ├── best.pt                # Helmet detection model
│   ├── number_plate.pt        # Number plate detection model
│   ├── numberplate.pt         # Alternate plate model
│   └── yolov8s.pt             # Vehicle detection model
│
├── helmetandnumberplate.py    # Helmet + number plate detection
├── speed_detection.py         # Speed estimation & violations
├── Bike_Helmet_Detection_model_training.ipynb
├── requirements.txt
└── README.md

