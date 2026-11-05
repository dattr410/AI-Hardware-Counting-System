# AI Hardware Counting System

An AI-powered hardware parts detection and counting system built with **YOLO26, Python, OpenCV, and Flask**. The application detects and counts small hardware components using a USB camera, providing an automated visual inspection workflow.

## Project Overview

This project focuses on automating the detection and counting of small hardware components commonly used in manufacturing and warehouse environments.

Using a custom-trained YOLO26 object detection model, the system identifies different hardware types and estimates their quantities from camera frames.

## Key Features

- **Real-time Object Detection:** Detect hardware components through a USB camera.
- **Multi-Class Recognition:** Supports three hardware categories:
  - Nut
  - Screw
  - Washer
- **Automated Counting:** Calculates the number of detected objects for each class.
- **Multi-Frame Inference:** Uses a 5-frame processing workflow to improve counting reliability.
- **Confidence Threshold:** Configurable detection confidence for each class.
- **Visual Results:** Displays bounding boxes, class labels, confidence scores, and counting results.
- **Web Interface:** Provides a Flask-based interface for interacting with the detection system.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLO26 | Object detection model |
| Ultralytics | Model training and inference |
| OpenCV | Camera processing and image handling |
| Flask | Backend and web application |
| HTML / CSS / JavaScript | User interface |
| PyTorch | Deep learning framework |

## System Architecture

```text
       USB Camera
            |
            v
      OpenCV Capture
            |
            v
       Frame Selection
            |
            v
      YOLO26 Inference
            |
            v
    Object Detection Results
            |
            v
       Object Counting
            |
            v
       Flask Backend
            |
            v
       Web Interface
            |
            v
   Detection and Count Display
```

## Supported Object Classes

| Class | Description |
|---|---|
| Nut | Hex nuts and other nut components |
| Screw | Screws and threaded fasteners |
| Washer | Flat washers |

## Project Structure

```text
AI-Hardware-Counting-System/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── best.pt
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   └── js/
│
└── output/
```

*Note: The structure above is illustrative. Actual files and directories may vary depending on the project configuration.*

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/dattr410/AI-Hardware-Counting-System.git
```

### 2. Navigate to the project directory

```bash
cd AI-Hardware-Counting-System
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the trained model

Place your trained YOLO26 model (`best.pt`) in the model directory.

Make sure the model path in `app.py` matches the actual location.

### 6. Run the application

```bash
python app.py
```

Open the local Flask address displayed in your terminal using a web browser.

## Model Information

| Configuration | Value |
|---|---|
| Model Family | YOLO26 |
| Model Size | YOLO26n |
| Task | Object Detection |
| Input Source | USB Camera |
| Number of Classes | 3 |
| Classes | Nut, Screw, Washer |

The model is trained on a custom hardware parts dataset.

## Development Challenges

Some of the main challenges addressed during development include:

- Distinguishing visually similar objects, especially small nuts and washers.
- Detecting small hardware components with different sizes.
- Handling overlapping and closely positioned objects.
- Reducing inconsistent predictions across consecutive frames.
- Improving detection reliability under changing lighting conditions.

## Future Improvements

- Improve classification accuracy for visually similar hardware components.
- Expand the number of supported hardware categories.
- Optimize inference speed for CPU-based systems.
- Improve performance under different lighting conditions.
- Explore instance segmentation for overlapping objects.
- Add detection history and reporting features.

## License

This project is intended for educational, research, and personal portfolio purposes.

---

**Developed as a Computer Vision and AI-based hardware counting project.**
