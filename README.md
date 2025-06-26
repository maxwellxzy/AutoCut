简体中文 | [English](./Readme_en.md)

# AutoCut - 基于DyHead和YOLOv11的错题自动切分系统

<div align="center">
    <img src="images\AutoCut_logo.svg" alt="logo" style="zoom:800%;" />
</div>


![AutoCut](https://img.shields.io/badge/AutoCut-错题检测系统-blue) ![Python](https://img.shields.io/badge/Python-3.8-green) ![Flask](https://img.shields.io/badge/Flask-2.0+-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red) ![YOLOv11](https://img.shields.io/badge/YOLOv11-Latest-yellow)

**基于DyHead和YOLOv11的错题自动切分系统**

## 📖 项目简介

AutoCut是一个集成了先进深度学习技术的错题自动检测系统，能够精准识别试卷中的题目区域和错误标记，具备以下核心能力：

- **[DyHead检测器](https://github.com/microsoft/DynamicHead)**: 基于动态头部检测技术的高精度题目区域分割模型
- **[YOLOv11模型](https://github.com/ultralytics/ultralytics)**: 最新的YOLO系列模型，用于错误符号精准检测
- **智能匹配算法**: 多策略错题-题目智能匹配，支持中心点包含、重叠面积、IOU、距离计算等方法
- **多类错误识别**: 支持5类错误符号检测（错号、斜线、半对、问号、圆圈）

## 🔧 核心技术

### 双模型检测架构
系统采用**双模型协同检测**技术实现高精度错题定位：

1. **题目区域检测**: 基于DyHead的DocumentRegionDetector实现精准题目分割
2. **错误符号识别**: 使用YOLOv11模型检测5类错误标记符号

### 智能匹配算法
系统实现了4层匹配策略，确保匹配的准确性和鲁棒性：

1. **中心点包含判断**: 优先级最高，错误符号中心点位于题目框内
2. **重叠面积计算**: 计算重叠区域占错误符号面积的比例
3. **IOU相似度**: 使用交并比算法评估框体相似度
4. **距离最近匹配**: 兜底策略，基于中心点距离进行匹配

## 🚀 快速开始

### 环境安装

```bash
pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e . 
pip install -e detectron2
pip install ultralytics
pip install flask
pip install pillow==9.5.0
```

### 模型文件准备

将以下模型文件存储于项目目录下：

- **DyHead配置文件**: `configs/dyhead_swint_atss_fpn_2x_ms.yaml`
- **DyHead权重文件**:`./cut_question.pth`，已上传至[百度网盘](https://pan.baidu.com/s/1RdSurxA_ohUmwOEy8Ld2ng)，提取码为`r5ht`
- **YOLOv11权重文件**: `./yolo_ckps/best.pt`，已上传至[百度网盘](https://pan.baidu.com/s/1RdSurxA_ohUmwOEy8Ld2ng)，提取码为`r5ht`

### 启动服务

```bash
python flask_error_detection.py
```

启动成功后访问 `http://localhost:5004` 使用Web界面进行错题检测。

## 📡 API接口

### 错题检测接口

**端点**: `POST /detect`

**请求参数**:

| 参数  | 类型 | 必填 | 说明                                           |
| ----- | ---- | ---- | ---------------------------------------------- |
| image | file | 是   | 试卷图片文件（支持 PNG/JPG/JPEG/BMP/TIFF/WebP） |

**请求示例**:

```bash
# 错题检测
curl -X POST http://localhost:5004/detect \
  -F "image=@test_paper.jpg"
```

**成功响应示例**:

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
        "match_method": "中心点包含"
      }
    ]
  }
}
```

**响应字段说明**:
- `error_count`: 检测到的错题数量
- `question_count`: 试卷总题目数量  
- `error_symbol_count`: 错误符号总数量
- `error_symbols_image`: 错误符号检测可视化图片（Base64）
- `questions_image`: 题目分割可视化图片（Base64）
- `matched_errors_image`: 错题匹配可视化图片（Base64）
- `error_details`: 详细的错题信息列表

## 📁 项目结构

```
AutoCut/
├── flask_error_detection.py  # Flask主应用程序
├── infer.py                  # 切题检测模块
├── requirements.txt          # Python依赖列表
├── configs/                  # 模型配置文件目录
│   └── dyhead_swint_atss_fpn_2x_ms.yaml
├── yolo_ckps/               # YOLO模型权重目录
│   └── best.pt
├── uploads/                 # 临时文件存储目录
├── outputs/                 # 输出结果目录
└── README.md               # 项目说明文档
```

## 🎨 检测结果展示

### Web界面展示
<div align="center">
    <img src="images\web_show.png" alt="web"/>
</div>

### demo

<div align="center">
    <img src="images\demo.png" alt="demo"/>
</div>