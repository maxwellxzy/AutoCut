"""
Improved Document Region Detection and Segmentation System
Based on Detectron2 and DyHead for intelligent document processing
"""
import os
import sys
import logging
import argparse
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import torch
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, default_setup
try:
    from dyhead import add_dyhead_config
    from extra import add_extra_config
except ImportError:
    logger.warning("DyHead modules not found. Please ensure they are installed.")


@dataclass
class DetectionBox:
    """Data class for detection boxes with all necessary information"""
    left: int
    top: int
    right: int
    bottom: int
    score: float
    vertical_id: Optional[int] = None
    
    @property
    def width(self) -> int:
        return self.right - self.left
    
    @property
    def height(self) -> int:
        return self.bottom - self.top
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)


class GeometryUtils:
    """Utility class for geometric calculations"""
    
    @staticmethod
    def calculate_iou(box1: DetectionBox, box2: DetectionBox) -> float:
        """Calculate Intersection over Union (IoU) between two boxes"""
        # Calculate intersection coordinates
        x1 = max(box1.left, box2.left)
        y1 = max(box1.top, box2.top)
        x2 = min(box1.right, box2.right)
        y2 = min(box1.bottom, box2.bottom)
        
        # No intersection case
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate areas
        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def has_horizontal_overlap(box1: DetectionBox, box2: DetectionBox, tolerance: int = 0) -> bool:
        """Check if two boxes overlap horizontally"""
        return not (box1.right + tolerance <= box2.left or box2.right + tolerance <= box1.left)
    
    @staticmethod
    def has_vertical_overlap(box1: DetectionBox, box2: DetectionBox, tolerance: int = 0) -> bool:
        """Check if two boxes overlap vertically"""
        return not (box1.bottom + tolerance <= box2.top or box2.bottom + tolerance <= box1.top)


class NMSProcessor:
    """Non-Maximum Suppression processor with detailed information tracking"""
    
    def __init__(self, iou_threshold: float = 0.2):
        self.iou_threshold = iou_threshold
    
    def process(self, boxes: List[DetectionBox]) -> Dict:
        """Apply NMS and return detailed information"""
        if not boxes:
            return {
                'kept_boxes': [],
                'removed_boxes': [],
                'iou_relationships': []
            }
        
        # Sort by score (descending)
        sorted_boxes = sorted(boxes, key=lambda x: x.score, reverse=True)
        
        kept_boxes = []
        removed_boxes = []
        iou_relationships = []
        
        for current_box in sorted_boxes:
            should_keep = True
            
            # Check overlap with kept boxes
            for kept_box in kept_boxes:
                iou = GeometryUtils.calculate_iou(current_box, kept_box)
                
                if iou > 0.01:  # Record meaningful relationships
                    iou_relationships.append({
                        'box1': current_box,
                        'box2': kept_box,
                        'iou': iou,
                        'suppressed': iou > self.iou_threshold
                    })
                
                if iou > self.iou_threshold:
                    should_keep = False
                    logger.debug(f"Suppressing box with score {current_box.score:.3f} "
                               f"(IoU={iou:.3f} > {self.iou_threshold})")
                    break
            
            if should_keep:
                kept_boxes.append(current_box)
            else:
                removed_boxes.append(current_box)
        
        logger.info(f"NMS: {len(boxes)} -> {len(kept_boxes)} boxes "
                   f"(threshold: {self.iou_threshold})")
        
        return {
            'kept_boxes': kept_boxes,
            'removed_boxes': removed_boxes,
            'iou_relationships': iou_relationships
        }


class LayoutAnalyzer:
    """Analyzer for detecting layout patterns and image center line"""
    
    def __init__(self):
        # Use fixed tolerance for internal layout analysis (not user-configurable)
        self.tolerance = 2
    
    def get_image_center_line(self, image_width: int) -> int:
        """Get the physical center line of the image"""
        return image_width // 2
    
    def box_crosses_center_line(self, box: DetectionBox, center_line: int) -> bool:
        """Check if a box crosses the center line"""
        return box.left < center_line < box.right
    
    def analyze_box_center_relationship(self, box: DetectionBox, center_line: int, slight_crossing_ratio: float = 0.1) -> Dict[str, any]:
        """Analyze the relationship between a box and the center line"""
        crosses = self.box_crosses_center_line(box, center_line)
        
        if not crosses:
            if box.right <= center_line:
                position = "left"
                distance_to_center = center_line - box.right
            elif box.left >= center_line:
                position = "right" 
                distance_to_center = box.left - center_line
            else:
                position = "ambiguous"
                distance_to_center = 0
            
            return {
                "crosses": False,
                "position": position,
                "distance_to_center": distance_to_center,
                "strategy": f"extend_to_center_line_from_{position}"
            }
        else:
            # Box crosses center line - determine if it's "slight" or "significant"
            left_distance = center_line - box.left
            right_distance = box.right - center_line
            box_width = box.width
            
            # The crossing distance is the smaller of the two distances
            crossing_distance = min(left_distance, right_distance)
            crossing_ratio = crossing_distance / box_width if box_width > 0 else 0
            
            # Determine if this is a "slight" crossing
            is_slight_crossing = crossing_ratio <= slight_crossing_ratio
            
            if left_distance > right_distance:
                main_side = "left"
                crossing_edge = "right"
            else:
                main_side = "right"
                crossing_edge = "left"
            
            if is_slight_crossing:
                strategy = f"slight_crossing_keep_{crossing_edge}_edge_extend_{main_side}"
            else:
                strategy = f"significant_crossing_use_original_strategy"
            
            return {
                "crosses": True,
                "is_slight_crossing": is_slight_crossing,
                "crossing_ratio": crossing_ratio,
                "crossing_distance": crossing_distance,
                "box_width": box_width,
                "main_side": main_side,
                "crossing_edge": crossing_edge,
                "left_distance": left_distance,
                "right_distance": right_distance,
                "strategy": strategy
            }
    
    def detect_left_right_layout(self, boxes: List[DetectionBox]) -> Tuple[bool, List[DetectionBox], List[DetectionBox], Optional[int]]:
        """Detect if boxes form a left-right layout pattern (for debugging/logging purposes)"""
        if len(boxes) < 2:
            return False, [], [], None
        
        logger.debug(f"Analyzing layout with {len(boxes)} boxes (internal tolerance: {self.tolerance}px)")
        
        # Build connectivity graph based on horizontal overlap
        components = self._find_connected_components(boxes)
        
        if len(components) == 2:
            # Check if the two components are truly left-right separated
            separated, left_group, right_group = self._validate_left_right_separation(
                boxes, components[0], components[1])
            
            if separated:
                center_line = self._calculate_center_line(left_group, right_group)
                logger.debug(f"Detected left-right layout: {len(left_group)} left, "
                           f"{len(right_group)} right boxes")
                return True, left_group, right_group, center_line
        
        logger.debug("Single column layout detected")
        return False, [], [], None
    
    def _find_connected_components(self, boxes: List[DetectionBox]) -> List[List[int]]:
        """Find connected components based on horizontal overlap"""
        n = len(boxes)
        visited = [False] * n
        adjacency = [[] for _ in range(n)]
        
        # Build adjacency list
        for i in range(n):
            for j in range(i + 1, n):
                if GeometryUtils.has_horizontal_overlap(boxes[i], boxes[j], self.tolerance):
                    adjacency[i].append(j)
                    adjacency[j].append(i)
        
        # DFS to find components
        components = []
        
        def dfs(node: int, current_component: List[int]):
            visited[node] = True
            current_component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    dfs(neighbor, current_component)
        
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, component)
                components.append(component)
        
        return components
    
    def _validate_left_right_separation(self, boxes: List[DetectionBox], 
                                      comp1: List[int], comp2: List[int]) -> Tuple[bool, List[DetectionBox], List[DetectionBox]]:
        """Validate that two components are truly left-right separated"""
        comp1_boxes = [boxes[i] for i in comp1]
        comp2_boxes = [boxes[i] for i in comp2]
        
        comp1_left = min(box.left for box in comp1_boxes)
        comp1_right = max(box.right for box in comp1_boxes)
        comp2_left = min(box.left for box in comp2_boxes)
        comp2_right = max(box.right for box in comp2_boxes)
        
        if comp1_right <= comp2_left:
            return True, comp1_boxes, comp2_boxes
        elif comp2_right <= comp1_left:
            return True, comp2_boxes, comp1_boxes
        else:
            return False, [], []
    
    def _calculate_center_line(self, left_group: List[DetectionBox], 
                             right_group: List[DetectionBox]) -> Optional[int]:
        """Calculate center line between left and right groups"""
        if not left_group or not right_group:
            return None
        
        left_rightmost = max(box.right for box in left_group)
        right_leftmost = min(box.left for box in right_group)
        
        return (left_rightmost + right_leftmost) // 2


class BoxProcessor:
    """Processor for extending and refining detection boxes"""
    
    def __init__(self, layout_analyzer: LayoutAnalyzer, slight_crossing_ratio: float = 0.1):
        self.layout_analyzer = layout_analyzer
        self.slight_crossing_ratio = slight_crossing_ratio
        logger.info(f"BoxProcessor initialized with slight crossing ratio: {slight_crossing_ratio*100}%")
    
    def process_boxes(self, boxes: List[DetectionBox], image_height: int, image_width: int) -> List[DetectionBox]:
        """Process boxes according to layout rules"""
        if not boxes:
            return boxes
        
        logger.info(f"Processing {len(boxes)} boxes...")
        
        # Sort boxes vertically and assign IDs
        sorted_boxes = self._sort_vertically(boxes)
        
        # Get image center line
        image_center_line = self.layout_analyzer.get_image_center_line(image_width)
        logger.info(f"Image center line: {image_center_line}")
        
        # Analyze layout (for backward compatibility and visualization)
        has_lr_layout, left_group, right_group, detected_center_line = \
            self.layout_analyzer.detect_left_right_layout(sorted_boxes)
        
        # Get global boundaries
        global_left = min(box.left for box in sorted_boxes)
        global_right = max(box.right for box in sorted_boxes)
        
        # Process each box
        processed_boxes = []
        
        for box in sorted_boxes:
            # Vertical extension
            extended_box = self._extend_vertically(box, sorted_boxes, image_height)
            
            # Horizontal extension with new center line strategy
            final_box = self._extend_horizontally_with_center_line_strategy(
                extended_box, sorted_boxes, global_left, global_right, image_center_line)
            
            processed_boxes.append(final_box)
        
        return processed_boxes
    
    def _sort_vertically(self, boxes: List[DetectionBox]) -> List[DetectionBox]:
        """Sort boxes vertically and assign IDs"""
        sorted_boxes = sorted(boxes, key=lambda x: x.top)
        for i, box in enumerate(sorted_boxes):
            box.vertical_id = i + 1
        return sorted_boxes
    
    def _extend_vertically(self, target_box: DetectionBox, 
                         all_boxes: List[DetectionBox], image_height: int) -> DetectionBox:
        """Extend box vertically if safe to do so"""
        # Check if bottom edge intersects with other boxes
        if self._has_bottom_intersection(target_box, all_boxes):
            logger.debug(f"Box {target_box.vertical_id}: No vertical extension (intersection)")
            return target_box
        
        # Find next box below
        next_box = self._find_next_box_below(target_box, all_boxes)
        
        new_bottom = next_box.top - 1 if next_box else image_height
        new_bottom = max(new_bottom, target_box.bottom)  # Don't shrink
        
        extended_box = DetectionBox(
            left=target_box.left,
            top=target_box.top,
            right=target_box.right,
            bottom=new_bottom,
            score=target_box.score,
            vertical_id=target_box.vertical_id
        )
        
        logger.debug(f"Box {target_box.vertical_id}: Vertical extension "
                    f"{target_box.bottom} -> {new_bottom}")
        
        return extended_box
    
    def _extend_horizontally_with_center_line_strategy(self, target_box: DetectionBox, all_boxes: List[DetectionBox],
                                                     global_left: int, global_right: int, image_center_line: int) -> DetectionBox:
        """Extend box horizontally using new center line strategy"""
        
        # Analyze the relationship between box and center line
        analysis = self.layout_analyzer.analyze_box_center_relationship(
            target_box, image_center_line, self.slight_crossing_ratio)
        
        logger.debug(f"Box {target_box.vertical_id} analysis: {analysis['strategy']}")
        
        if not analysis["crosses"]:
            # Box doesn't cross center line
            if analysis["position"] == "left":
                # Box is completely on the left side, extend right to center line
                new_left = self._calculate_safe_extension(
                    target_box, all_boxes, 'left', global_left, None)
                new_right = self._calculate_safe_extension(
                    target_box, all_boxes, 'right', image_center_line, None)
                logger.debug(f"Box {target_box.vertical_id} completely on left side, extending right to center line {image_center_line}")
                
            elif analysis["position"] == "right":
                # Box is completely on the right side, extend left to center line
                new_left = self._calculate_safe_extension(
                    target_box, all_boxes, 'left', image_center_line, None)
                new_right = self._calculate_safe_extension(
                    target_box, all_boxes, 'right', global_right, None)
                logger.debug(f"Box {target_box.vertical_id} completely on right side, extending left to center line {image_center_line}")
                
            else:
                # Ambiguous position, keep original size
                new_left = target_box.left
                new_right = target_box.right
                logger.warning(f"Box {target_box.vertical_id} has ambiguous position, keeping original size")
                
        else:
            # Box crosses center line
            crossing_distance = analysis["crossing_distance"]
            crossing_ratio = analysis["crossing_ratio"]
            box_width = analysis["box_width"]
            is_slight = analysis["is_slight_crossing"]
            
            logger.debug(f"Box {target_box.vertical_id} crosses center line: "
                        f"distance={crossing_distance}px, ratio={crossing_ratio:.2%}, "
                        f"box_width={box_width}px, is_slight={is_slight}")
            
            if is_slight:
                # Slight crossing: keep the crossing edge unchanged, extend the other edge
                main_side = analysis["main_side"]
                crossing_edge = analysis["crossing_edge"]
                
                if crossing_edge == "right":
                    # Right edge crosses center slightly, keep it unchanged, extend left edge
                    new_left = self._calculate_safe_extension(
                        target_box, all_boxes, 'left', global_left, None)
                    new_right = target_box.right  # Keep right edge unchanged
                    logger.debug(f"Box {target_box.vertical_id} slight right crossing ({crossing_ratio:.1%}) - keeping right edge unchanged")
                    
                elif crossing_edge == "left":
                    # Left edge crosses center slightly, keep it unchanged, extend right edge
                    new_left = target_box.left  # Keep left edge unchanged
                    new_right = self._calculate_safe_extension(
                        target_box, all_boxes, 'right', global_right, None)
                    logger.debug(f"Box {target_box.vertical_id} slight left crossing ({crossing_ratio:.1%}) - keeping left edge unchanged")
                    
                else:
                    # Fallback case
                    new_left = target_box.left
                    new_right = target_box.right
                    logger.warning(f"Box {target_box.vertical_id} unexpected crossing pattern, keeping original size")
            else:
                # Significant crossing: use original strategy (extend to global boundaries)
                new_left = self._calculate_safe_extension(
                    target_box, all_boxes, 'left', global_left, None)
                new_right = self._calculate_safe_extension(
                    target_box, all_boxes, 'right', global_right, None)
                logger.debug(f"Box {target_box.vertical_id} significant crossing ({crossing_ratio:.1%}) - using original strategy")
        
        extended_box = DetectionBox(
            left=new_left,
            top=target_box.top,
            right=new_right,
            bottom=target_box.bottom,
            score=target_box.score,
            vertical_id=target_box.vertical_id
        )
        
        logger.debug(f"Box {target_box.vertical_id}: Horizontal extension "
                    f"[{target_box.left}, {target_box.right}] -> [{new_left}, {new_right}]")
        
        return extended_box
    
    def _extend_horizontally(self, target_box: DetectionBox, all_boxes: List[DetectionBox],
                           global_left: int, global_right: int, center_line: Optional[int]) -> DetectionBox:
        """Extend box horizontally considering center line constraints (original method for backward compatibility)"""
        new_left = self._calculate_safe_extension(
            target_box, all_boxes, 'left', global_left, center_line)
        new_right = self._calculate_safe_extension(
            target_box, all_boxes, 'right', global_right, center_line)
        
        extended_box = DetectionBox(
            left=new_left,
            top=target_box.top,
            right=new_right,
            bottom=target_box.bottom,
            score=target_box.score,
            vertical_id=target_box.vertical_id
        )
        
        return extended_box
    
    def _has_bottom_intersection(self, target_box: DetectionBox, 
                               all_boxes: List[DetectionBox]) -> bool:
        """Check if target box's bottom edge intersects with other boxes"""
        for other_box in all_boxes:
            if other_box == target_box:
                continue
            
            # Check horizontal overlap
            if GeometryUtils.has_horizontal_overlap(target_box, other_box):
                # Check if bottom edge intersects vertically
                if other_box.top <= target_box.bottom <= other_box.bottom:
                    return True
        return False
    
    def _find_next_box_below(self, target_box: DetectionBox, 
                           all_boxes: List[DetectionBox]) -> Optional[DetectionBox]:
        """Find the nearest box below the target box"""
        candidates = []
        
        for other_box in all_boxes:
            if (other_box != target_box and 
                other_box.top > target_box.bottom and
                GeometryUtils.has_horizontal_overlap(target_box, other_box)):
                
                candidates.append((other_box.top - target_box.bottom, other_box))
        
        return min(candidates, key=lambda x: x[0])[1] if candidates else None
    
    def _calculate_safe_extension(self, target_box: DetectionBox, all_boxes: List[DetectionBox],
                                direction: str, boundary: int, center_line: Optional[int]) -> int:
        """Calculate safe extension position considering collisions and center line"""
        if direction == 'left':
            target_pos = boundary
            current_pos = target_box.left
        else:  # right
            target_pos = boundary
            current_pos = target_box.right
        
        # Check center line constraint
        if center_line is not None:
            if direction == 'left' and target_pos < center_line:
                target_pos = max(target_pos, center_line)
            elif direction == 'right' and target_pos > center_line:
                target_pos = min(target_pos, center_line)
        
        # Check collisions with other boxes
        for other_box in all_boxes:
            if other_box == target_box:
                continue
            
            if GeometryUtils.has_vertical_overlap(target_box, other_box):
                if direction == 'left':
                    if (other_box.right > target_pos and 
                        other_box.right < current_pos):
                        target_pos = max(target_pos, other_box.right)
                else:  # right
                    if (other_box.left < target_pos and 
                        other_box.left > current_pos):
                        target_pos = min(target_pos, other_box.left)
        
        return target_pos


class DocumentRegionDetector:
    """Main class for document region detection"""
    
    def __init__(self, config_path: str = "configs/dyhead_swint_atss_fpn_2x_ms.yaml",
                 confidence_threshold: float = 0.25, nms_threshold: float = 0.2,
                 slight_crossing_ratio: float = 0.1):
        
        self.confidence_threshold = confidence_threshold
        self.nms_processor = NMSProcessor(nms_threshold)
        self.layout_analyzer = LayoutAnalyzer()
        self.box_processor = BoxProcessor(self.layout_analyzer, slight_crossing_ratio)
        
        # Initialize model
        self.predictor = self._setup_model(config_path)
        
        # Storage for analysis results
        self.last_nms_info = None
        self.last_center_line = None
        
        logger.info(f"Initialized detector - confidence: {confidence_threshold}, "
                   f"NMS: {nms_threshold}, slight crossing ratio: {slight_crossing_ratio*100}%")
    
    def _setup_model(self, config_path: str) -> DefaultPredictor:
        """Setup the detection model"""
        cfg = get_cfg()
        add_dyhead_config(cfg)
        add_extra_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.freeze()
        
        # Setup with dummy args
        class Args:
            def __init__(self):
                self.config = config_path
                self.num_gpus = 1

        logger.info(f"Model device: {cfg.MODEL.DEVICE}")

        default_setup(cfg, Args())
        return DefaultPredictor(cfg)
    
    def detect(self, image: np.ndarray) -> List[DetectionBox]:
        """Main detection method"""
        # Run detection
        outputs = self.predictor(image)
        instances = outputs["instances"]
        confident_detections = instances[instances.scores > self.confidence_threshold]
        
        # Convert to DetectionBox objects
        boxes = []
        if len(confident_detections) > 0:
            boxes_tensor = confident_detections.pred_boxes
            scores_tensor = confident_detections.scores
            
            for i, (box_tensor, score_tensor) in enumerate(zip(boxes_tensor, scores_tensor)):
                coords = [int(x) for x in box_tensor.cpu().tolist()]
                score = float(score_tensor.cpu().item())
                
                if len(coords) == 4:
                    boxes.append(DetectionBox(
                        left=coords[0], top=coords[1],
                        right=coords[2], bottom=coords[3],
                        score=score
                    ))
        
        logger.info(f"Initial detection: {len(boxes)} boxes")
        
        # Apply NMS
        self.last_nms_info = self.nms_processor.process(boxes)
        nms_boxes = self.last_nms_info['kept_boxes']
        
        # Process boxes with layout rules (including image dimensions)
        image_height, image_width = image.shape[:2]
        processed_boxes = self.box_processor.process_boxes(nms_boxes, image_height, image_width)
        
        # Store center line for visualization (using image center line now)
        self.last_center_line = self.layout_analyzer.get_image_center_line(image_width)
        
        logger.info(f"Final result: {len(processed_boxes)} processed boxes")
        logger.info(f"Using image center line: {self.last_center_line}")
        return processed_boxes


class Visualizer:
    """Visualization utilities for detection results"""
    
    @staticmethod
    def draw_detection_results(image: np.ndarray, original_boxes: List[DetectionBox],
                             processed_boxes: List[DetectionBox], center_line: Optional[int] = None) -> np.ndarray:
        """Draw detection results with annotations"""
        result_image = image.copy()
        
        # Draw original boxes (red dashed)
        for box in original_boxes:
            Visualizer._draw_dashed_rectangle(
                result_image, (box.left, box.top), (box.right, box.bottom), (0, 0, 255), 2)
        
        # Draw processed boxes (blue solid)
        for box in processed_boxes:
            cv2.rectangle(result_image, (box.left, box.top), (box.right, box.bottom), (255, 0, 0), 2)
            
            # Add label
            label = f"{box.vertical_id}:{box.score:.3f}"
            label_pos = (box.left, box.top - 10 if box.top > 20 else box.top + 20)
            
            # Draw text background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(result_image, 
                         (label_pos[0], label_pos[1] - text_height - baseline),
                         (label_pos[0] + text_width, label_pos[1] + baseline),
                         (200, 0, 0), -1)
            cv2.putText(result_image, label, label_pos, font, font_scale, (255, 255, 255), thickness)
        
        # Draw global boundaries (green dashed)
        if processed_boxes:
            global_left = min(box.left for box in processed_boxes)
            global_right = max(box.right for box in processed_boxes)
            image_height = image.shape[0]
            
            Visualizer._draw_dashed_line(result_image, (global_left, 0), (global_left, image_height), (0, 255, 0), 2)
            Visualizer._draw_dashed_line(result_image, (global_right, 0), (global_right, image_height), (0, 255, 0), 2)
        
        # Draw image center line (purple solid)
        if center_line is not None:
            image_height = image.shape[0]
            cv2.line(result_image, (center_line, 0), (center_line, image_height), (255, 0, 255), 3)
            cv2.putText(result_image, f"Image Center: {center_line}", (center_line + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Add legend
        legend_lines = [
            "Legend: Red Dashed=Original, Blue Solid=Processed",
            "Green Dashed=Global Boundaries",
            "Purple Solid=Image Center Line"
        ]
        
        for i, line in enumerate(legend_lines):
            cv2.putText(result_image, line, (10, 25 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    @staticmethod
    def _draw_dashed_rectangle(image: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                             color: Tuple[int, int, int], thickness: int):
        """Draw dashed rectangle"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        Visualizer._draw_dashed_line(image, (x1, y1), (x2, y1), color, thickness)
        Visualizer._draw_dashed_line(image, (x2, y1), (x2, y2), color, thickness)
        Visualizer._draw_dashed_line(image, (x2, y2), (x1, y2), color, thickness)
        Visualizer._draw_dashed_line(image, (x1, y2), (x1, y1), color, thickness)
    
    @staticmethod
    def _draw_dashed_line(image: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                        color: Tuple[int, int, int], thickness: int, dash_length: int = 10):
        """Draw dashed line"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length == 0:
            return
        
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        
        current_length = 0
        while current_length < length:
            start_x = int(x1 + dx * current_length)
            start_y = int(y1 + dy * current_length)
            
            end_length = min(current_length + dash_length, length)
            end_x = int(x1 + dx * end_length)
            end_y = int(y1 + dy * end_length)
            
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
            current_length += dash_length * 2


def process_image(image_path: str, output_dir: str, detector: DocumentRegionDetector) -> bool:
    """Process a single image"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Cannot read image: {image_path}")
            return False
        
        # Detect regions
        processed_boxes = detector.detect(image)
        
        # Visualize results
        nms_boxes = detector.last_nms_info['kept_boxes']
        center_line = detector.last_center_line
        
        result_image = Visualizer.draw_detection_results(
            image, nms_boxes, processed_boxes, center_line)
        
        # Save result
        input_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"processed_{input_filename}")
        cv2.imwrite(output_path, result_image)
        
        logger.info(f"Processed: {image_path} -> {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Document Region Detection System")
    parser.add_argument("--input", "-i", default="./images", help="Input image path or directory")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--config", "-c", default="configs/dyhead_swint_atss_fpn_2x_ms.yaml", 
                       help="Model config path")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.2, help="NMS IoU threshold")
    parser.add_argument("--slight_crossing_ratio", type=float, default=0.1, 
                       help="Ratio threshold for slight crossing detection (default: 0.1 = 10%%)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector
    try:
        detector = DocumentRegionDetector(
            config_path=args.config,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms_threshold,
            slight_crossing_ratio=args.slight_crossing_ratio
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return
    
    # Process input
    if os.path.isfile(args.input):
        process_image(args.input, args.output, detector)
    elif os.path.isdir(args.input):
        # Process all images in directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        processed_count = 0
        failed_count = 0
        
        for filename in os.listdir(args.input):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(args.input, filename)
                if process_image(image_path, args.output, detector):
                    processed_count += 1
                else:
                    failed_count += 1
        
        logger.info(f"Batch processing completed: {processed_count} successful, {failed_count} failed")
    else:
        logger.error(f"Input path does not exist: {args.input}")


if __name__ == "__main__":
    main()