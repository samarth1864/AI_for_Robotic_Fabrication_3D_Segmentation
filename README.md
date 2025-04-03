# AI_for_Robotic_Fabrication_3D_Segmentation

# YOLO-3D for Fabrication Environments

This project adapts [niconielsen32/YOLO-3D](https://github.com/niconielsen32/YOLO-3D) for real-time 3D object detection in fabrication spaces. It combines YOLOv11 for object detection with Depth Anything v2 for depth estimation, enabling 3D bounding box visualization and Bird's Eye View (BEV).

## Purpose

**What:** Locate different elements and agents in a fabrication environment  
**Why:** Improve safety, monitoring, and spatial control

## Features

- Real-time object detection (YOLOv11)
- Depth estimation (Depth Anything v2)
- Pseudo-3D and Bird’s Eye View visualization
- Object tracking
- Webcam or video input

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV, NumPy
- `pip install -r requirements.txt`

## Usage

```bash
git clone https://github.com/your-username/YOLO-3D-Fabrication.git
cd YOLO-3D-Fabrication
python run.py
