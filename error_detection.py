"""
错题检测与定位系统
整合切题模型和错号检测模型，自动识别错题并切割保存
"""
import os
import sys
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import random

# 添加当前目录到路径，以便导入其他模块
sys.path.append('.')

# 导入切题模型相关模块
from infer import DocumentRegionDetector, DetectionBox

# 导入错号检测模型
from ultralytics import YOLO

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ErrorBox:
    """错号检测框数据类"""
    left: int
    top: int
    right: int
    bottom: int
    confidence: float
    class_id: int = 0  # 添加类别ID
    class_name: str = ""  # 添加类别名称
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)
    
    @property
    def area(self) -> int:
        return (self.right - self.left) * (self.bottom - self.top)


@dataclass
class MatchResult:
    """匹配结果数据类"""
    question_box: DetectionBox
    error_boxes: List[ErrorBox]
    match_method: str
    match_confidence: float


class ColorGenerator:
    """颜色生成器，为不同的匹配对生成不同颜色"""
    
    def __init__(self):
        # 预定义一些鲜明的颜色（BGR格式）
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
            # 如果预定义颜色用完，随机生成颜色
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
    def draw_match_results(image: np.ndarray, match_results: List[MatchResult]) -> np.ndarray:
        """在图像上绘制匹配结果"""
        result_image = image.copy()
        color_generator = ColorGenerator()
        
        # 为每个匹配结果分配颜色并绘制
        for i, result in enumerate(match_results):
            color = color_generator.get_next_color()
            
            # 绘制题目框
            q_box = result.question_box
            cv2.rectangle(result_image, 
                         (q_box.left, q_box.top), 
                         (q_box.right, q_box.bottom), 
                         color, 3)
            
            # 绘制题目标签
            question_label = f"题目{q_box.vertical_id}"
            label_pos = (q_box.left, q_box.top - 10 if q_box.top > 30 else q_box.top + 25)
            
            # 绘制文字背景
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
            for j, error_box in enumerate(result.error_boxes):
                # 错号框用同样的颜色，但稍微细一点的线条
                cv2.rectangle(result_image,
                             (error_box.left, error_box.top),
                             (error_box.right, error_box.bottom),
                             color, 2)
                
                # 绘制错号标签，包含类别信息
                error_label = f"{error_box.class_name}({error_box.confidence:.2f})"
                error_label_pos = (error_box.left, 
                                 error_box.top - 5 if error_box.top > 20 else error_box.bottom + 20)
                
                # 绘制错号文字背景
                (text_width, text_height), baseline = cv2.getTextSize(
                    error_label, font, 0.5, 1)
                
                cv2.rectangle(result_image,
                             (error_label_pos[0], error_label_pos[1] - text_height - baseline),
                             (error_label_pos[0] + text_width, error_label_pos[1] + baseline),
                             color, -1)
                
                cv2.putText(result_image, error_label, error_label_pos,
                           font, 0.5, (255, 255, 255), 1)
        
        # 添加图例
        result_image = ResultVisualizer._add_legend(result_image, match_results)
        
        return result_image
    
    @staticmethod
    def _add_legend(image: np.ndarray, match_results: List[MatchResult]) -> np.ndarray:
        """添加图例说明"""
        if not match_results:
            return image
        
        # 在图像右上角添加图例
        legend_start_x = image.shape[1] - 350
        legend_start_y = 30
        legend_height = 25
        
        # 绘制图例背景
        legend_bg_height = len(match_results) * legend_height + 40
        cv2.rectangle(image, 
                     (legend_start_x - 10, legend_start_y - 20),
                     (image.shape[1] - 10, legend_start_y + legend_bg_height),
                     (0, 0, 0), -1)
        
        # 绘制图例标题
        cv2.putText(image, "错题对应关系:", 
                   (legend_start_x, legend_start_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 重新生成颜色（保持一致）
        color_generator = ColorGenerator()
        
        for i, result in enumerate(match_results):
            color = color_generator.get_next_color()
            y_pos = legend_start_y + (i + 1) * legend_height
            
            # 绘制颜色块
            cv2.rectangle(image,
                         (legend_start_x, y_pos - 8),
                         (legend_start_x + 20, y_pos + 8),
                         color, -1)
            
            # 绘制说明文字，包含错号类别信息
            error_types = set(e.class_name for e in result.error_boxes)
            error_type_str = ",".join(error_types)
            legend_text = f"题目{result.question_box.vertical_id} ({error_type_str})"
            cv2.putText(image, legend_text,
                       (legend_start_x + 30, y_pos + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image


class BoxMatcher:
    """检测框匹配器"""
    
    def __init__(self, overlap_threshold: float = 0.3, iou_threshold: float = 0.1, 
                 max_distance_ratio: float = 0.5):
        self.overlap_threshold = overlap_threshold  # 重叠面积阈值
        self.iou_threshold = iou_threshold          # IOU阈值
        self.max_distance_ratio = max_distance_ratio # 最大距离比例
    
    def match_errors_to_questions(self, question_boxes: List[DetectionBox], 
                                error_boxes: List[ErrorBox]) -> List[MatchResult]:
        """将错号框匹配到题目框"""
        results = []
        unmatched_errors = error_boxes.copy()
        
        # 为每个错号找到最佳匹配的题目
        for error_box in error_boxes:
            best_match = self._find_best_match(error_box, question_boxes)
            if best_match:
                question_box, method, confidence = best_match
                
                # 查找是否已有该题目的匹配结果
                existing_result = None
                for result in results:
                    if result.question_box.vertical_id == question_box.vertical_id:
                        existing_result = result
                        break
                
                if existing_result:
                    # 添加到现有结果
                    existing_result.error_boxes.append(error_box)
                else:
                    # 创建新的匹配结果
                    results.append(MatchResult(
                        question_box=question_box,
                        error_boxes=[error_box],
                        match_method=method,
                        match_confidence=confidence
                    ))
                
                # 从未匹配列表中移除
                if error_box in unmatched_errors:
                    unmatched_errors.remove(error_box)
        
        logger.info(f"匹配完成: {len(results)}个错题, {len(unmatched_errors)}个未匹配错号")
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
        
        # 计算图像大小用于距离阈值
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
            # 距离越近，置信度越高
            confidence = max(0.1, 1.0 - (best_distance_value / max_distance))
            return best_distance, "距离最近", confidence
        
        # 如果所有策略都失败，选择距离最近的作为兜底
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
            return 1000  # 默认值
        
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
        
        # 初始化切题模型
        logger.info("初始化切题模型...")
        self.qieti_detector = DocumentRegionDetector(config_path=qieti_config_path)
        
        # 初始化错号检测模型
        logger.info("初始化错号检测模型...")
        self.yolo_model = YOLO(yolo_model_path)
        
        # 定义各类别的置信度阈值
        self.class_thresholds = {
            0: 0.4,   # cuo
            1: 0.5,   # xie
            2: 0.6,   # bandui
            3: 0.6,   # wenhao
            4: 0.4    # yuanquan
        }
        
        # 定义类别名称映射
        self.class_names = {
            0: "cuo",
            1: "xie", 
            2: "bandui",
            3: "wenhao",
            4: "yuanquan"
        }
        
        # 初始化匹配器
        self.matcher = BoxMatcher()
        
        logger.info("错题检测系统初始化完成")
        logger.info(f"类别置信度阈值: {self.class_thresholds}")
    
    def detect_error_questions(self, image_path: str) -> Tuple[List[MatchResult], np.ndarray]:
        """检测错题"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        logger.info(f"开始处理图像: {image_path}")
        
        # 1. 切题检测
        logger.info("执行切题检测...")
        question_boxes = self.qieti_detector.detect(image)
        logger.info(f"检测到 {len(question_boxes)} 个题目区域")
        
        # 2. 错号检测
        logger.info("执行错号检测...")
        yolo_results = self.yolo_model.predict(
            source=image_path,
            conf=0.01,  # 设置很低的初始阈值，后续根据类别过滤
            iou=0.5,    # 保持原有IOU阈值
            save=False,
            verbose=False,
            device=0,
        )
        
        error_boxes = self._parse_yolo_results(yolo_results)
        logger.info(f"检测到 {len(error_boxes)} 个错号")
        
        # 3. 匹配错号到题目
        logger.info("匹配错号到题目...")
        match_results = self.matcher.match_errors_to_questions(question_boxes, error_boxes)
        
        return match_results, image
    
    def _parse_yolo_results(self, yolo_results) -> List[ErrorBox]:
        """解析YOLO检测结果并根据类别过滤置信度"""
        error_boxes = []
        
        for result in yolo_results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # 获取坐标
                confidences = result.boxes.conf.cpu().numpy()  # 获取置信度
                class_ids = result.boxes.cls.cpu().numpy()  # 获取类别ID
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    cls_id = int(cls_id)
                    
                    # 根据类别ID获取对应的置信度阈值
                    threshold = self.class_thresholds.get(cls_id, 0.4)  # 默认阈值0.4
                    
                    # 只保留置信度超过该类别阈值的检测结果
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
                        
                        logger.debug(f"检测到错号: {class_name}, 置信度: {conf:.3f}, 阈值: {threshold}")
        
        # 按类别统计检测结果
        class_counts = {}
        for error_box in error_boxes:
            class_counts[error_box.class_name] = class_counts.get(error_box.class_name, 0) + 1
        
        logger.info(f"各类别检测统计: {class_counts}")
        
        return error_boxes
    
    def save_error_questions(self, image: np.ndarray, match_results: List[MatchResult], 
                           output_dir: str, base_filename: str):
        """保存错题检测可视化结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        if match_results:
            # 有错题时，保存可视化结果
            visual_result = ResultVisualizer.draw_match_results(image, match_results)
            visual_filename = f"{base_filename}_错题检测结果.jpg"
            visual_output_path = os.path.join(output_dir, visual_filename)
            cv2.imwrite(visual_output_path, visual_result)
            logger.info(f"保存可视化结果: {visual_output_path}")
            
            # 打印详细的匹配信息
            for i, result in enumerate(match_results):
                q_box = result.question_box
                logger.info(f"题目ID: {q_box.vertical_id}, "
                           f"匹配方式: {result.match_method}, "
                           f"置信度: {result.match_confidence:.3f}, "
                           f"错号数量: {len(result.error_boxes)}")
                
                for j, error_box in enumerate(result.error_boxes):
                    logger.info(f"  错号{j+1}: {error_box.class_name}, "
                               f"置信度: {error_box.confidence:.3f}")
        else:
            # 没有错题时，保存原图像
            original_filename = f"{base_filename}_无错题.jpg"
            original_output_path = os.path.join(output_dir, original_filename)
            cv2.imwrite(original_output_path, image)
            logger.info(f"未检测到错题，保存原图像: {original_output_path}")


def process_single_image(image_path: str, detector: ErrorQuestionDetector, output_dir: str):
    """处理单张图像"""
    try:
        start_time = time.time()
        
        # 检测错题
        match_results, image = detector.detect_error_questions(image_path)
        
        # 保存结果（无论是否有错题都保存）
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        detector.save_error_questions(image, match_results, output_dir, base_filename)
        
        # 输出结果摘要
        total_time = time.time() - start_time
        
        if match_results:
            logger.info(f"处理完成 - 用时: {total_time:.2f}秒, 检测到错题: {len(match_results)}个")
            logger.info(f"结果已保存到: {output_dir}")
            logger.info(f"  - 可视化标注图像: 1张 ({base_filename}_错题检测结果.jpg)")
            
            # 输出详细匹配信息
            print("\n=== 检测结果 ===")
            print(f"共检测到 {len(match_results)} 个错题，每个错题和其对应的错号在可视化图像中用相同颜色标出")
            print("-" * 60)
            for i, result in enumerate(match_results, 1):
                q_box = result.question_box
                print(f"错题 {i} (题目ID: {q_box.vertical_id}):")
                print(f"  题目框: [{q_box.left}, {q_box.top}, {q_box.right}, {q_box.bottom}]")
                print(f"  错号框及类别: ", end="")
                for j, e_box in enumerate(result.error_boxes):
                    print(f"[{e_box.left}, {e_box.top}, {e_box.right}, {e_box.bottom}]({e_box.class_name}:{e_box.confidence:.2f})", end="")
                    if j < len(result.error_boxes) - 1:
                        print(", ", end="")
                print(f"\n  匹配方式: {result.match_method}")
                print(f"  匹配置信度: {result.match_confidence:.3f}")
                print(f"  颜色标识: 在可视化图像中，此错题和错号用第{i}种颜色标出")
                print()
        else:
            logger.info(f"处理完成 - 用时: {total_time:.2f}秒, 未检测到错题")
            logger.info(f"原图像已保存到: {output_dir}")
            logger.info(f"  - 原图像: 1张 ({base_filename}_无错题.jpg)")
            
            print("\n=== 检测结果 ===")
            print("本图像中未检测到错题，已保存原图像")
        
    except Exception as e:
        logger.error(f"处理图像 {image_path} 时出错: {str(e)}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="错题检测与定位系统")
    parser.add_argument("--input", "-i", default="./images", 
                       help="输入图像路径或目录")
    parser.add_argument("--output", "-o", default="./output", 
                       help="输出目录")
    parser.add_argument("--qieti_config", default="configs/dyhead_swint_atss_fpn_2x_ms.yaml",
                       help="切题模型配置文件路径")
    parser.add_argument("--yolo_model", default="./yolo_ckps/best.pt",
                       help="YOLO模型权重路径")
    
    args = parser.parse_args()
    
    # 初始化检测器
    try:
        detector = ErrorQuestionDetector(
            qieti_config_path=args.qieti_config,
            yolo_model_path=args.yolo_model
        )
    except Exception as e:
        logger.error(f"初始化检测器失败: {e}")
        return
    
    # 处理输入
    if os.path.isfile(args.input):
        # 处理单张图像
        process_single_image(args.input, detector, args.output)
    elif os.path.isdir(args.input):
        # 处理文件夹中的所有图像
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in os.listdir(args.input) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        logger.info(f"找到 {len(image_files)} 张图像")
        
        for filename in image_files:
            image_path = os.path.join(args.input, filename)
            print(f"\n{'='*50}")
            print(f"处理图像: {filename}")
            print('='*50)
            process_single_image(image_path, detector, args.output)
    else:
        logger.error(f"输入路径不存在: {args.input}")


if __name__ == "__main__":
    main()