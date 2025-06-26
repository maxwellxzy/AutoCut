[简体中文](./README.md) | English

# AutoCut - Automatic Error Question Segmentation System Based on DyHead and YOLOv11

<div align="center">
    <img src="images\AutoCut_logo.svg" alt="logo" style="zoom:800%;" />
</div>

![AutoCut](https://img.shields.io/badge/AutoCut-Error%20Detection%20System-blue) ![Python](https://img.shields.io/badge/Python-3.8-green) ![Flask](https://img.shields.io/badge/Flask-2.0+-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red) ![YOLOv11](https://img.shields.io/badge/YOLOv11-Latest-yellow)

**Automatic Error Question Segmentation System Based on DyHead and YOLOv11**

## 📖 Project Overview

AutoCut is an advanced error question detection system that integrates cutting-edge deep learning technologies. It can accurately identify question regions and error marks in test papers, featuring the following core capabilities:

- **[DyHead Detector](https://github.com/microsoft/DynamicHead)**: High-precision question region segmentation model based on dynamic head detection technology
- **[YOLOv11 Model](https://github.com/ultralytics/ultralytics)**: Latest YOLO series model for precise error symbol detection
- **Intelligent Matching Algorithm**: Multi-strategy error-question intelligent matching supporting center point containment, overlap area, IOU, distance calculation, and other methods
- **Multi-class Error Recognition**: Supports detection of 5 types of error symbols (cross, diagonal, half-correct, question mark, circle)

## 🔧 Core Technologies

### Dual-Model Detection Architecture
The system adopts **dual-model collaborative detection** technology to achieve high-precision error question localization:

1. **Question Region Detection**: Implements precise question segmentation using DyHead-based DocumentRegionDetector
2. **Error Symbol Recognition**: Uses YOLOv11 model to detect 5 types of error marking symbols

### Intelligent Matching Algorithm
The system implements a 4-layer matching strategy to ensure matching accuracy and robustness:

1. **Center Point Containment**: Highest priority, error symbol center point located within question box
2. **Overlap Area Calculation**: Calculates the proportion of overlapping area to error symbol area
3. **IOU Similarity**: Uses Intersection over Union algorithm to evaluate box similarity
4. **Nearest Distance Matching**: Fallback strategy based on center point distance matching

## 🚀 Quick Start

### Environment Setup

```bash
pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e . 
pip install -e detectron2
pip install ultralytics
pip install flask
pip install pillow==9.5.0
```

### Model File Preparation

Store the following model files in the project directory:

- **DyHead Configuration File**: `configs/dyhead_swint_atss_fpn_2x_ms.yaml`
- **DyHead Weight File**: `./cut_question.pth`, uploaded to [Baidu Drive](https://pan.baidu.com/s/1RdSurxA_ohUmwOEy8Ld2ng), extraction code: `r5ht`
- **YOLOv11 Weight File**: `./yolo_ckps/best.pt`, uploaded to [Baidu Drive](https://pan.baidu.com/s/1RdSurxA_ohUmwOEy8Ld2ng), extraction code: `r5ht`

### Start Service

```bash
python flask_error_detection.py
```

After successful startup, visit `http://localhost:5004` to use the web interface for error question detection.

## 📡 API Interface

### Error Question Detection Interface

**Endpoint**: `POST /detect`

**Request Parameters**:

| Parameter | Type | Required | Description                                                |
| --------- | ---- | -------- | ---------------------------------------------------------- |
| image     | file | Yes      | Test paper image file (supports PNG/JPG/JPEG/BMP/TIFF/WebP) |

**Request Example**:

```bash
# Error question detection
curl -X POST http://localhost:5004/detect \
  -F "image=@test_paper.jpg"
```

**Success Response Example**:

```json
{
  "success": true,
  "data": {
    "error_count": 3,
    "question_count": 10,
    "error_symbol_count": 5,
    "error_symbols_image": "data:image/jpeg;base64,/9j/4AAQ...",
    "questions_image": "data:image/jpeg;base64,/9j/4AAQ...",
    "matched_errors_image": "data:image/jpeg;base64,/9j/4AAQ...",
    "error_details": [
      {
        "question_box": {
          "bbox": [100, 50, 200, 150]
        },
        "error_boxes": [
          {
            "bbox": [120, 70, 30, 25],
            "confidence": 0.85,
            "class_name": "cuo"
          }
        ],
        "match_method": "Center Point Containment"
      }
    ]
  }
}
```

**Response Field Description**:
- `error_count`: Number of detected error questions
- `question_count`: Total number of questions in the test paper
- `error_symbol_count`: Total number of error symbols
- `error_symbols_image`: Error symbol detection visualization image (Base64)
- `questions_image`: Question segmentation visualization image (Base64)
- `matched_errors_image`: Error question matching visualization image (Base64)
- `error_details`: Detailed error question information list

## 📁 Project Structure

```
AutoCut/
├── flask_error_detection.py  # Flask main application
├── infer.py                  # Question segmentation detection module
├── requirements.txt          # Python dependency list
├── configs/                  # Model configuration file directory
│   └── dyhead_swint_atss_fpn_2x_ms.yaml
├── yolo_ckps/               # YOLO model weight directory
│   └── best.pt
├── uploads/                 # Temporary file storage directory
├── outputs/                 # Output result directory
└── README.md               # Project documentation
```

## 🎨 Detection Results Display

### Web Interface Display
<div align="center">
    <img src="images\web_show.png" alt="web"/>
</div>

### Demo

<div align="center">
    <img src="images\demo.png" alt="demo"/>
</div>