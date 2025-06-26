"""
Flask错题检测与定位Web服务
提供Web界面和API接口用于错题检测
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import cv2
import numpy as np
import logging
import json
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import time
import random
import uuid

from flask import Flask, render_template_string, request, jsonify, send_file
from werkzeug.utils import secure_filename

# 添加当前目录到路径，以便导入其他模块
sys.path.append('.')

# 导入切题模型相关模块
from infer import DocumentRegionDetector, DetectionBox

# 导入错号检测模型
from ultralytics import YOLO
import torch

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# 确保上传和输出目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


@dataclass
class ErrorBox:
    """错号检测框数据类"""
    left: int
    top: int
    right: int
    bottom: int
    confidence: float
    class_id: int = 0
    class_name: str = ""

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    @property
    def area(self) -> int:
        return (self.right - self.left) * (self.bottom - self.top)


@dataclass
class MatchResult:
    """匹配结果数据类"""
    question_box: Dict
    error_boxes: List[Dict]
    match_method: str


class ColorGenerator:
    """颜色生成器，为不同的匹配对生成不同颜色"""

    def __init__(self):
        self.colors = [
            (0, 0, 255),    # 红色
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 255, 255),  # 黄色
            (255, 0, 255),  # 紫色
            (255, 255, 0),  # 青色
            (128, 0, 255),  # 橙色
            (0, 165, 255),  # 橙红色
            (147, 20, 255), # 深粉色
            (0, 140, 255),  # 深橙色
            (128, 128, 0),  # 橄榄色
            (128, 0, 128),  # 紫红色
            (0, 128, 128),  # 青绿色
            (255, 192, 203), # 粉色
            (32, 178, 170),  # 浅海绿色
        ]
        self.color_index = 0

    def get_next_color(self) -> Tuple[int, int, int]:
        """获取下一个颜色"""
        if self.color_index < len(self.colors):
            color = self.colors[self.color_index]
        else:
            color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
        self.color_index += 1
        return color


class ResultVisualizer:
    """结果可视化器"""

    @staticmethod
    def draw_question_boxes(image: np.ndarray, question_boxes: List[DetectionBox]) -> np.ndarray:
        """绘制题目框"""
        result_image = image.copy()

        for i, q_box in enumerate(question_boxes):
            # 绘制题目框
            cv2.rectangle(result_image,
                         (q_box.left, q_box.top),
                         (q_box.right, q_box.bottom),
                         (0, 255, 0), 2)  # 绿色

            # 绘制题目标签
            question_label = f"题目{i+1}"
            label_pos = (q_box.left, q_box.top - 10 if q_box.top > 30 else q_box.top + 25)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                question_label, font, font_scale, thickness)

            cv2.rectangle(result_image,
                         (label_pos[0], label_pos[1] - text_height - baseline),
                         (label_pos[0] + text_width, label_pos[1] + baseline),
                         (0, 255, 0), -1)

            cv2.putText(result_image, question_label, label_pos,
                       font, font_scale, (255, 255, 255), thickness)

        return result_image

    @staticmethod
    def draw_error_boxes(image: np.ndarray, error_boxes: List[ErrorBox]) -> np.ndarray:
        """绘制错号框"""
        result_image = image.copy()

        for error_box in error_boxes:
            # 绘制错号框
            cv2.rectangle(result_image,
                         (error_box.left, error_box.top),
                         (error_box.right, error_box.bottom),
                         (0, 0, 255), 2)  # 红色

            # 绘制错号标签
            error_label = f"{error_box.class_name}({error_box.confidence:.2f})"
            error_label_pos = (error_box.left,
                             error_box.top - 5 if error_box.top > 20 else error_box.bottom + 20)

            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(
                error_label, font, 0.5, 1)

            cv2.rectangle(result_image,
                         (error_label_pos[0], error_label_pos[1] - text_height - baseline),
                         (error_label_pos[0] + text_width, error_label_pos[1] + baseline),
                         (0, 0, 255), -1)

            cv2.putText(result_image, error_label, error_label_pos,
                       font, 0.5, (255, 255, 255), 1)

        return result_image

    @staticmethod
    def draw_match_results(image: np.ndarray, match_results: List[MatchResult]) -> np.ndarray:
        """在图像上绘制匹配结果"""
        result_image = image.copy()
        color_generator = ColorGenerator()

        for i, result in enumerate(match_results):
            color = color_generator.get_next_color()

            # 使用bbox格式的坐标
            q_dict = result.question_box
            q_bbox = q_dict['bbox']  # [x, y, width, height]
            q_left, q_top = q_bbox[0], q_bbox[1]
            q_right, q_bottom = q_bbox[0] + q_bbox[2], q_bbox[1] + q_bbox[3]

            # 绘制题目框
            cv2.rectangle(result_image,
                         (q_left, q_top),
                         (q_right, q_bottom),
                         color, 3)

            # 绘制题目标签
            question_label = f"题目{i+1}"
            label_pos = (q_left, q_top - 10 if q_top > 30 else q_top + 25)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                question_label, font, font_scale, thickness)

            cv2.rectangle(result_image,
                         (label_pos[0], label_pos[1] - text_height - baseline),
                         (label_pos[0] + text_width, label_pos[1] + baseline),
                         color, -1)

            cv2.putText(result_image, question_label, label_pos,
                       font, font_scale, (255, 255, 255), thickness)

            # 绘制对应的错号框
            for j, error_dict in enumerate(result.error_boxes):
                e_bbox = error_dict['bbox']  # [x, y, width, height]
                e_left, e_top = e_bbox[0], e_bbox[1]
                e_right, e_bottom = e_bbox[0] + e_bbox[2], e_bbox[1] + e_bbox[3]

                cv2.rectangle(result_image,
                             (e_left, e_top),
                             (e_right, e_bottom),
                             color, 2)

                error_label = f"{error_dict['class_name']}({error_dict['confidence']:.2f})"
                error_label_pos = (e_left,
                                 e_top - 5 if e_top > 20 else e_bottom + 20)

                (text_width, text_height), baseline = cv2.getTextSize(
                    error_label, font, 0.5, 1)

                cv2.rectangle(result_image,
                             (error_label_pos[0], error_label_pos[1] - text_height - baseline),
                             (error_label_pos[0] + text_width, error_label_pos[1] + baseline),
                             color, -1)

                cv2.putText(result_image, error_label, error_label_pos,
                           font, 0.5, (255, 255, 255), 1)

        return result_image


class BoxMatcher:
    """检测框匹配器"""

    def __init__(self, overlap_threshold: float = 0.3, iou_threshold: float = 0.1,
                 max_distance_ratio: float = 0.5):
        self.overlap_threshold = overlap_threshold
        self.iou_threshold = iou_threshold
        self.max_distance_ratio = max_distance_ratio

    def match_errors_to_questions(self, question_boxes: List[DetectionBox],
                                error_boxes: List[ErrorBox]) -> List[MatchResult]:
        """将错号框匹配到题目框"""
        results = []
        unmatched_errors = error_boxes.copy()

        for error_box in error_boxes:
            best_match = self._find_best_match(error_box, question_boxes)
            if best_match:
                question_box, method, confidence = best_match

                existing_result = None
                for result in results:
                    # 通过bbox坐标比较来找到对应的结果
                    result_bbox = result.question_box['bbox']
                    if (result_bbox[0] == question_box.left and
                        result_bbox[1] == question_box.top and
                        result_bbox[2] == question_box.right - question_box.left and
                        result_bbox[3] == question_box.bottom - question_box.top):
                        existing_result = result
                        break

                # 构建error_dict，使用bbox格式，去掉class_id
                error_dict = {
                    'bbox': [error_box.left, error_box.top,
                            error_box.right - error_box.left, error_box.bottom - error_box.top],
                    'confidence': error_box.confidence,
                    'class_name': error_box.class_name
                }

                if existing_result:
                    existing_result.error_boxes.append(error_dict)
                else:
                    # 构建question_dict，使用bbox格式，去掉vertical_id
                    question_dict = {
                        'bbox': [question_box.left, question_box.top,
                                question_box.right - question_box.left, question_box.bottom - question_box.top]
                    }

                    results.append(MatchResult(
                        question_box=question_dict,
                        error_boxes=[error_dict],
                        match_method=method
                    ))

                if error_box in unmatched_errors:
                    unmatched_errors.remove(error_box)

        return results

    def _find_best_match(self, error_box: ErrorBox,
                        question_boxes: List[DetectionBox]) -> Optional[Tuple[DetectionBox, str, float]]:
        """为单个错号框找到最佳匹配的题目框"""

        # 策略1: 中心点包含判断
        for q_box in question_boxes:
            if self._point_in_box(error_box.center, q_box):
                return q_box, "中心点包含", 1.0

        # 策略2: 重叠面积判断
        best_overlap = None
        best_overlap_ratio = 0

        for q_box in question_boxes:
            overlap_area = self._calculate_overlap_area(error_box, q_box)
            if overlap_area > 0:
                overlap_ratio = overlap_area / error_box.area
                if overlap_ratio > best_overlap_ratio and overlap_ratio >= self.overlap_threshold:
                    best_overlap = q_box
                    best_overlap_ratio = overlap_ratio

        if best_overlap:
            return best_overlap, "重叠面积", best_overlap_ratio

        # 策略3: IOU判断
        best_iou = None
        best_iou_value = 0

        for q_box in question_boxes:
            iou = self._calculate_iou(error_box, q_box)
            if iou > best_iou_value and iou >= self.iou_threshold:
                best_iou = q_box
                best_iou_value = iou

        if best_iou:
            return best_iou, "IOU", best_iou_value

        # 策略4: 距离最近判断
        if not question_boxes:
            return None

        image_diagonal = self._estimate_image_diagonal(question_boxes)
        max_distance = image_diagonal * self.max_distance_ratio

        best_distance = None
        best_distance_value = float('inf')

        for q_box in question_boxes:
            distance = self._calculate_center_distance(error_box, q_box)
            if distance < best_distance_value and distance <= max_distance:
                best_distance = q_box
                best_distance_value = distance

        if best_distance:
            confidence = max(0.1, 1.0 - (best_distance_value / max_distance))
            return best_distance, "距离最近", confidence

        closest_box = min(question_boxes,
                         key=lambda q: self._calculate_center_distance(error_box, q))
        distance = self._calculate_center_distance(error_box, closest_box)
        confidence = max(0.05, 1.0 - (distance / image_diagonal))
        return closest_box, "兜底匹配", confidence

    def _point_in_box(self, point: Tuple[int, int], box: DetectionBox) -> bool:
        """判断点是否在框内"""
        x, y = point
        return box.left <= x <= box.right and box.top <= y <= box.bottom

    def _calculate_overlap_area(self, error_box: ErrorBox, question_box: DetectionBox) -> int:
        """计算两个框的重叠面积"""
        x1 = max(error_box.left, question_box.left)
        y1 = max(error_box.top, question_box.top)
        x2 = min(error_box.right, question_box.right)
        y2 = min(error_box.bottom, question_box.bottom)

        if x2 <= x1 or y2 <= y1:
            return 0

        return (x2 - x1) * (y2 - y1)

    def _calculate_iou(self, error_box: ErrorBox, question_box: DetectionBox) -> float:
        """计算IOU"""
        overlap_area = self._calculate_overlap_area(error_box, question_box)
        if overlap_area == 0:
            return 0.0

        error_area = error_box.area
        question_area = question_box.area
        union_area = error_area + question_area - overlap_area

        return overlap_area / union_area if union_area > 0 else 0.0

    def _calculate_center_distance(self, error_box: ErrorBox, question_box: DetectionBox) -> float:
        """计算两个框中心点的距离"""
        ex, ey = error_box.center
        qx, qy = question_box.center
        return ((ex - qx) ** 2 + (ey - qy) ** 2) ** 0.5

    def _estimate_image_diagonal(self, question_boxes: List[DetectionBox]) -> float:
        """估算图像对角线长度"""
        if not question_boxes:
            return 1000

        min_x = min(box.left for box in question_boxes)
        max_x = max(box.right for box in question_boxes)
        min_y = min(box.top for box in question_boxes)
        max_y = max(box.bottom for box in question_boxes)

        return ((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5


class ErrorQuestionDetector:
    """错题检测主类"""

    def __init__(self,
                 qieti_config_path: str = "configs/dyhead_swint_atss_fpn_2x_ms.yaml",
                 yolo_model_path: str = "./yolo_ckps/best.pt"):

        logger.info("初始化切题模型...")
        self.qieti_detector = DocumentRegionDetector(config_path=qieti_config_path)

        logger.info("初始化错号检测模型...")
        self.yolo_model = YOLO(yolo_model_path)

        self.class_thresholds = {
            0: 0.4,   # cuo
            1: 0.5,   # xie
            2: 0.6,   # bandui
            3: 0.6,   # wenhao
            4: 0.4    # yuanquan
        }

        self.class_names = {
            0: "cuo",
            1: "xie",
            2: "bandui",
            3: "wenhao",
            4: "yuanquan"
        }

        self.matcher = BoxMatcher()
        logger.info("错题检测系统初始化完成")

    def detect_error_questions(self, image: np.ndarray) -> Tuple[List[MatchResult], List[DetectionBox], List[ErrorBox]]:
        """检测错题，返回匹配结果、题目框和错号框"""

        logger.info("开始执行错题检测...")

        # 1. 切题检测
        logger.info("执行切题检测...")
        question_boxes = self.qieti_detector.detect(image)
        logger.info(f"检测到 {len(question_boxes)} 个题目区域")

        # 2. 错号检测
        logger.info("执行错号检测...")
        yolo_results = self.yolo_model.predict(
            source=image,
            conf=0.01,
            iou=0.5,
            save=False,
            verbose=False,
            device=0,
        )

        error_boxes = self._parse_yolo_results(yolo_results)
        logger.info(f"检测到 {len(error_boxes)} 个错号")

        # 3. 匹配错号到题目
        logger.info("匹配错号到题目...")
        match_results = self.matcher.match_errors_to_questions(question_boxes, error_boxes)
        logger.info(f"匹配完成，共 {len(match_results)} 个错题")

        return match_results, question_boxes, error_boxes

    def _parse_yolo_results(self, yolo_results) -> List[ErrorBox]:
        """解析YOLO检测结果并根据类别过滤置信度"""
        error_boxes = []

        for result in yolo_results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    cls_id = int(cls_id)
                    threshold = self.class_thresholds.get(cls_id, 0.4)

                    if conf >= threshold:
                        class_name = self.class_names.get(cls_id, f"class_{cls_id}")

                        error_boxes.append(ErrorBox(
                            left=int(box[0]),
                            top=int(box[1]),
                            right=int(box[2]),
                            bottom=int(box[3]),
                            confidence=float(conf),
                            class_id=cls_id,
                            class_name=class_name
                        ))

        return error_boxes


# 全局检测器实例
detector = None

def init_detector():
    """初始化检测器"""
    global detector
    if detector is None:
        try:
            logger.info("正在初始化错题检测器...")

            # 初始化前清理CUDA缓存
            if torch.cuda.is_available():
                with torch.cuda.device("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                logger.info("初始化前CUDA缓存已清理")

            detector = ErrorQuestionDetector()
            logger.info("检测器初始化成功")

            # 初始化后再次清理CUDA缓存
            if torch.cuda.is_available():
                with torch.cuda.device("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                logger.info("初始化后CUDA缓存已清理")

        except Exception as e:
            logger.error(f"检测器初始化失败: {e}")
            logger.error(f"详细错误信息: {str(e)}")

            # 初始化失败时也要清理缓存
            if torch.cuda.is_available():
                with torch.cuda.device("cuda"):
                    torch.cuda.empty_cache()

            raise Exception(f"初始化检测器失败: {str(e)}")


def image_to_base64(image: np.ndarray) -> str:
    """将OpenCV图像转换为base64字符串"""
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{img_str}"


@app.route('/detect', methods=['POST'])
def detect():
    """检测接口"""
    try:
        logger.info("收到检测请求")

        # 检查是否有文件上传
        if 'image' not in request.files:
            logger.warning("请求中没有图片文件")
            return jsonify({'success': False, 'message': '没有上传图片文件'})

        file = request.files['image']
        if file.filename == '':
            logger.warning("没有选择文件")
            return jsonify({'success': False, 'message': '没有选择文件'})

        logger.info(f"接收到文件: {file.filename}")

        # 检查文件类型
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
        if not ('.' in file.filename and
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            logger.warning(f"不支持的文件格式: {file.filename}")
            return jsonify({'success': False, 'message': '不支持的文件格式'})

        # 读取图片
        logger.info("正在读取图片...")
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("无法解析图片文件")
            return jsonify({'success': False, 'message': '无法解析图片文件'})

        logger.info(f"图片尺寸: {image.shape}")

        # 初始化检测器（如果还没有初始化）
        if detector is None:
            logger.info("检测器未初始化，正在初始化...")
            init_detector()

        # 执行检测
        logger.info("开始执行检测...")
        match_results, question_boxes, error_boxes = detector.detect_error_questions(image)

        logger.info("正在生成可视化图片...")
        # 生成可视化图片
        # 1. 错误符号检测图
        error_symbols_image = ResultVisualizer.draw_error_boxes(image, error_boxes)
        error_symbols_b64 = image_to_base64(error_symbols_image)

        # 2. 题目分割图
        questions_image = ResultVisualizer.draw_question_boxes(image, question_boxes)
        questions_b64 = image_to_base64(questions_image)

        # 3. 错题匹配图
        matched_errors_image = ResultVisualizer.draw_match_results(image, match_results)
        matched_errors_b64 = image_to_base64(matched_errors_image)

        logger.info("构建响应数据...")
        # 构建响应数据
        response_data = {
            'error_count': len(match_results),
            'question_count': len(question_boxes),
            'error_symbol_count': len(error_boxes),
            'error_symbols_image': error_symbols_b64,
            'questions_image': questions_b64,
            'matched_errors_image': matched_errors_b64,
            'error_details': [asdict(result) for result in match_results]
        }

        logger.info(f"检测完成，返回结果: 错题{len(match_results)}个，题目{len(question_boxes)}个，错号{len(error_boxes)}个")

        return jsonify({
            'success': True,
            'data': response_data
        })

    except Exception as e:
        logger.error(f"检测过程中出错: {str(e)}")
        import traceback
        logger.error(f"详细错误堆栈: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': f'检测失败: {str(e)}'
        })


@app.route('/', methods=['GET'])
def index():
    """主页面"""
    html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>错题检测系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .upload-section {
            padding: 40px;
            text-align: center;
        }

        .upload-area {
            border: 3px dashed #4facfe;
            border-radius: 15px;
            padding: 40px;
            margin: 20px 0;
            background: #f8f9ff;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .upload-area:hover {
            border-color: #00f2fe;
            background: #f0f4ff;
            transform: translateY(-2px);
        }

        .upload-area.dragover {
            border-color: #00f2fe;
            background: #e8f4ff;
            transform: scale(1.02);
        }

        .upload-icon {
            font-size: 4em;
            color: #4facfe;
            margin-bottom: 20px;
        }

        .upload-text {
            font-size: 1.3em;
            color: #666;
            margin-bottom: 15px;
        }

        .upload-hint {
            color: #999;
            font-size: 0.9em;
        }

        #fileInput {
            display: none;
        }

        .btn {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 10px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(79, 172, 254, 0.4);
        }

        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4facfe;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .results {
            display: none;
            padding: 40px;
            border-top: 2px solid #e9ecef;
            margin-top: 20px;
        }

        .results-header {
            text-align: center;
            margin-bottom: 30px;
        }

        .results-header h2 {
            color: #333;
            font-size: 2em;
            margin-bottom: 10px;
        }

        .stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }

        .stat-item {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            min-width: 150px;
        }

        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            display: block;
        }

        .stat-label {
            font-size: 1em;
            opacity: 0.9;
        }

        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .image-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }

        .image-card:hover {
            transform: translateY(-5px);
        }

        .image-card h3 {
            color: #333;
            margin-bottom: 15px;
            text-align: center;
            font-size: 1.3em;
        }

        .image-card img {
            width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            cursor: pointer;
            transition: transform 0.3s ease;
        }

        .image-card img:hover {
            transform: scale(1.05);
        }

        /* 图片放大模态框样式 */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            animation: fadeIn 0.3s;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .modal-content {
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            margin-top: 5%;
            border-radius: 10px;
            animation: zoomIn 0.3s;
        }

        @keyframes zoomIn {
            from { transform: scale(0.5); }
            to { transform: scale(1); }
        }

        .modal-close {
            position: absolute;
            top: 20px;
            right: 35px;
            color: #fff;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.3s;
        }

        .modal-close:hover {
            color: #ff6b6b;
        }

        .modal-caption {
            margin: auto;
            display: block;
            width: 80%;
            max-width: 700px;
            text-align: center;
            color: #ccc;
            padding: 10px 0;
            font-size: 1.2em;
        }

        .error-list {
            margin-top: 30px;
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
        }

        .error-list h3 {
            color: #333;
            margin-bottom: 15px;
            text-align: center;
        }

        .error-item {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #4facfe;
        }

        .error-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }

        .error-details {
            color: #666;
            font-size: 0.9em;
            line-height: 1.4;
        }

        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 15px;
            }

            .header {
                padding: 20px;
            }

            .header h1 {
                font-size: 2em;
            }

            .upload-section {
                padding: 20px;
            }

            .stats {
                gap: 15px;
            }

            .stat-item {
                min-width: 120px;
                padding: 15px;
            }

            .image-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 智能错题检测系统</h1>
            <p>上传试卷图片，自动识别错题位置和错误标记</p>
        </div>

        <div class="upload-section">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📁</div>
                <div class="upload-text">点击选择图片文件</div>
                <div class="upload-hint">支持 JPG, PNG, JPEG 格式，最大 16MB</div>
            </div>
            <input type="file" id="fileInput" accept="image/*">
            <button class="btn" id="uploadBtn" onclick="uploadImage()" disabled>🚀 开始检测</button>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>正在处理图片，请稍候...</p>
        </div>

        <div class="results" id="results">
            <div class="results-header">
                <h2>🎉 检测结果</h2>
                <div class="stats" id="stats"></div>
            </div>

            <div class="image-grid" id="imageGrid"></div>

            <div class="error-list" id="errorList"></div>
        </div>

        <!-- 图片放大模态框 -->
        <div id="imageModal" class="modal">
            <span class="modal-close" onclick="closeModal()">&times;</span>
            <img class="modal-content" id="modalImage">
            <div class="modal-caption" id="modalCaption"></div>
        </div>
    </div>

    <script>
        let selectedFile = null;

        // 文件选择处理
        document.getElementById('fileInput').addEventListener('change', function(e) {
            selectedFile = e.target.files[0];
            if (selectedFile) {
                document.getElementById('uploadBtn').disabled = false;
                updateUploadArea();
            }
        });

        // 拖拽功能
        const uploadArea = document.querySelector('.upload-area');

        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                selectedFile = files[0];
                document.getElementById('fileInput').files = files;
                document.getElementById('uploadBtn').disabled = false;
                updateUploadArea();
            }
        });

        function updateUploadArea() {
            const uploadText = document.querySelector('.upload-text');
            const uploadHint = document.querySelector('.upload-hint');

            if (selectedFile) {
                uploadText.textContent = `已选择: ${selectedFile.name}`;
                uploadHint.textContent = `文件大小: ${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`;
            }
        }

        async function uploadImage() {
            if (!selectedFile) {
                alert('请先选择图片文件');
                return;
            }

            // 显示加载状态
            document.getElementById('loading').style.display = 'block';
            // 隐藏之前的结果
            document.getElementById('results').style.display = 'none';

            const formData = new FormData();
            formData.append('image', selectedFile);

            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();

                if (result.success) {
                    displayResults(result.data);
                } else {
                    throw new Error(result.message || '检测失败');
                }
            } catch (error) {
                console.error('Error:', error);
                alert(`检测失败: ${error.message}`);
                // 出错时隐藏加载状态，但保持上传区域可见
                document.getElementById('loading').style.display = 'none';
                resetUploadArea();
            }
        }

        function displayResults(data) {
            // 隐藏加载状态
            document.getElementById('loading').style.display = 'none';

            // 显示统计信息（交换错题符号和错题数量的位置）
            const statsHtml = `
                <div class="stat-item">
                    <span class="stat-number">${data.error_symbol_count}</span>
                    <span class="stat-label">错题符号</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">${data.error_count}</span>
                    <span class="stat-label">错题数量</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">${data.question_count}</span>
                    <span class="stat-label">题目总数</span>
                </div>
            `;
            document.getElementById('stats').innerHTML = statsHtml;

            // 显示图片（添加点击放大功能）
            const imageGridHtml = `
                <div class="image-card">
                    <h3>🔍 错误符号检测</h3>
                    <img src="${data.error_symbols_image}" alt="错误符号检测结果" onclick="openModal(this, '错误符号检测结果')">
                </div>
                <div class="image-card">
                    <h3>📝 题目分割</h3>
                    <img src="${data.questions_image}" alt="题目分割结果" onclick="openModal(this, '题目分割结果')">
                </div>
                <div class="image-card">
                    <h3>❌ 错题匹配</h3>
                    <img src="${data.matched_errors_image}" alt="错题匹配结果" onclick="openModal(this, '错题匹配结果')">
                </div>
            `;
            document.getElementById('imageGrid').innerHTML = imageGridHtml;

            // 显示错题详情
            if (data.error_details && data.error_details.length > 0) {
                let errorListHtml = '<h3>📋 错题详细信息</h3>';
                data.error_details.forEach((error, index) => {
                    const errorSymbols = error.error_boxes.map(box => 
                        `${box.class_name}(${(box.confidence * 100).toFixed(1)}%)`
                    ).join(', ');

                    // 使用bbox格式显示坐标
                    const bbox = error.question_box.bbox;
                    const bboxStr = `[${bbox[0]}, ${bbox[1]}, ${bbox[2]}, ${bbox[3]}]`;

                    errorListHtml += `
                        <div class="error-item">
                            <div class="error-title">错题 ${index + 1}</div>
                            <div class="error-details">
                                <strong>题目区域 (x,y,width,height):</strong> ${bboxStr}<br>
                                <strong>错误符号:</strong> ${errorSymbols}<br>
                                <strong>匹配方式:</strong> ${error.match_method}
                            </div>
                        </div>
                    `;
                });
                document.getElementById('errorList').innerHTML = errorListHtml;
            } else {
                document.getElementById('errorList').innerHTML = '<h3>🎉 恭喜！未检测到错题</h3>';
            }

            // 显示结果区域，但保持上传区域可见
            document.getElementById('results').style.display = 'block';

            // 重置上传区域状态，允许继续上传
            resetUploadArea();
        }

        // 图片放大功能
        function openModal(img, caption) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            const modalCaption = document.getElementById('modalCaption');

            modal.style.display = 'block';
            modalImg.src = img.src;
            modalCaption.innerHTML = caption;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }

        // 点击模态框背景关闭
        document.getElementById('imageModal').addEventListener('click', function(e) {
            if (e.target === this) {
                closeModal();
            }
        });

        // ESC键关闭模态框
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeModal();
            }
        });

        function resetUploadArea() {
            selectedFile = null;
            document.getElementById('fileInput').value = '';
            document.getElementById('uploadBtn').disabled = true;
            document.querySelector('.upload-text').textContent = '点击选择图片文件';
            document.querySelector('.upload-hint').textContent = '支持 JPG, PNG, JPEG 格式，最大 16MB';
        }

        function resetPage() {
            resetUploadArea();
            document.querySelector('.upload-section').style.display = 'block';
            document.getElementById('loading').style.display = 'none';
            document.getElementById('results').style.display = 'none';
        }
    </script>
</body>
</html>
    """
    return render_template_string(html_template)

if __name__ == '__main__':
    try:
        # 初始化检测器
        init_detector()
        logger.info("启动Flask服务器...")
        app.run(host='0.0.0.0', port=5004, debug=False)
    except Exception as e:
        logger.error(f"启动服务器失败: {e}")