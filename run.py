#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
import torch
import simpleaudio as sa  # For alarm sound
from pathlib import Path
from math import sqrt

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import our modules for depth estimation and 3D utilities
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from load_camera_params import load_camera_params, apply_camera_params_to_estimator

############################################
# Updated RoboflowDetector using new API key/model_id
############################################
from inference_sdk import InferenceHTTPClient

class RoboflowDetector:
    def __init__(self, model_id="greyscaleimages/2"):
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="Xn0fVaXqUOR6gYmVVKib"
        )
        self.model_id = model_id
        # Update class mapping: Adjust based on your model's classes.
        # For example, if your model returns "robot" for robot detections:
        self.class_mapping = {"robot": 0, "person": 1}
    
    def detect(self, image, track=False):
        """
        Save the current image temporarily, send it to Roboflow, and return the
        original image along with a list of detections. Each detection is a tuple:
        (bbox, score, class_id, obj_id), where bbox is [xmin, ymin, xmax, ymax].
        """
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, image)
        result = self.client.infer(temp_path, model_id=self.model_id)
        os.remove(temp_path)
        
        detections = []
        if "predictions" in result:
            for pred in result["predictions"]:
                # Roboflow returns the center coordinates (x, y) plus width and height.
                x_center = pred.get("x")
                y_center = pred.get("y")
                w = pred.get("width")
                h = pred.get("height")
                xmin = x_center - w / 2
                ymin = y_center - h / 2
                xmax = x_center + w / 2
                ymax = y_center + h / 2
                bbox = [xmin, ymin, xmax, ymax]
                score = float(pred.get("confidence", 0))
                class_name = pred.get("class", "").lower()
                # Use the mapping; if not found, return -1.
                class_id = self.class_mapping.get(class_name, -1)
                obj_id = None  # Tracking is not implemented here.
                detections.append((bbox, score, class_id, obj_id))
        return image, detections

    def get_class_names(self):
        # Return a mapping from class id to class name.
        return {v: k for k, v in self.class_mapping.items()}

############################################
# End of RoboflowDetector definition
############################################

# Define a function to play an alarm sound.
def play_alarm():
    try:
        # Ensure "alarm.wav" is located in D:/IAAC/Robotic_Fabrication/
        wave_obj = sa.WaveObject.from_wave_file("D:/IAAC/Robotic_Fabrication/alarm.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print("Error playing alarm:", e)

def main():
    """Main function for processing video with 3D segmentation and alarm."""
    # ===============================================
    # Input/Output Settings
    source = r"D:\IAAC\Robotic_Fabrication\data\lab_video.mp4"  # Change to 0 for webcam if needed.
    output_path = "output.mp4"  # Output video file path.
    
    # Initialize the RoboflowDetector with the updated model id.
    detector = RoboflowDetector(model_id="greyscaleimages/2")
    
    depth_model_size = "small"  # Depth model size: "small", "base", "large"
    
    # Device settings.
    device = 'cpu'  # Use CPU (change to 'cuda' if available and desired).
    
    # Feature toggles.
    enable_tracking = True
    enable_bev = True  # Enable Bird's Eye View visualization.
    enable_pseudo_3d = True  # Enable pseudo-3D visualization (if applicable).
    
    # Camera parameters (if any)
    camera_params_file = None  # Use default parameters if None.
    # ===============================================
    
    print(f"Using device: {device}")
    print("Initializing models...")
    
    # Initialize Depth Estimator.
    try:
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device=device
        )
    except Exception as e:
        print(f"Error initializing depth estimator: {e}")
        print("Falling back to CPU for depth estimation")
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device='cpu'
        )
    
    # Initialize 3D bounding box estimator.
    bbox3d_estimator = BBox3DEstimator()
    
    # Initialize Bird's Eye View if enabled.
    if enable_bev:
        bev = BirdEyeView(scale=60, size=(300, 300))
    
    # Open video source.
    try:
        if isinstance(source, str) and source.isdigit():
            source = int(source)
    except ValueError:
        pass
    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"
    
    # Alarm parameters.
    alarm_threshold = 50  # Pixel distance threshold.
    
    print("Starting processing...")
    while True:
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            print("Exiting program...")
            break
        
        try:
            ret, frame = cap.read()
            if not ret:
                break
            
            original_frame = frame.copy()
            detection_frame = frame.copy()
            result_frame = frame.copy()
            
            # Step 1: Object Detection using RoboflowDetector.
            try:
                detection_frame, detections = detector.detect(detection_frame, track=enable_tracking)
            except Exception as e:
                print(f"Error during object detection: {e}")
                detections = []
                cv2.putText(detection_frame, "Detection Error", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Step 2: Depth Estimation.
            try:
                depth_map = depth_estimator.estimate_depth(original_frame)
                depth_colored = depth_estimator.colorize_depth(depth_map)
            except Exception as e:
                print(f"Error during depth estimation: {e}")
                depth_map = np.zeros((height, width), dtype=np.float32)
                depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(depth_colored, "Depth Error", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Step 3: 3D Bounding Box Estimation & Proximity Check.
            boxes_3d = []
            active_ids = []
            human_centers = []  # For storing centers of persons.
            robot_centers = []  # For storing centers of robots.
            
            for detection in detections:
                try:
                    bbox, score, class_id, obj_id = detection
                    class_name = detector.get_class_names()[class_id]
                    
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)
                    
                    # Collect center points based on class.
                    if 'person' in class_name.lower():
                        human_centers.append((center_x, center_y))
                    elif 'robot' in class_name.lower():
                        robot_centers.append((center_x, center_y))
                    
                    if class_name.lower() in ['person']:
                        depth_value = depth_estimator.get_depth_at_point(depth_map, center_x, center_y)
                        depth_method = 'center'
                    else:
                        depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                        depth_method = 'median'
                    
                    box_3d = {
                        'bbox_2d': bbox,
                        'depth_value': depth_value,
                        'depth_method': depth_method,
                        'class_name': class_name,
                        'object_id': obj_id,
                        'score': score
                    }
                    boxes_3d.append(box_3d)
                    if obj_id is not None:
                        active_ids.append(obj_id)
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            
            bbox3d_estimator.cleanup_trackers(active_ids)
            
            # Proximity Check: Trigger alarm if any human is too close to any robot.
            alarm_triggered = False
            for hp in human_centers:
                for rp in robot_centers:
                    dx = hp[0] - rp[0]
                    dy = hp[1] - rp[1]
                    dist = sqrt(dx * dx + dy * dy)
                    if dist < alarm_threshold:
                        alarm_triggered = True
                        break
                if alarm_triggered:
                    break
            if alarm_triggered:
                print("ALARM: Human is too close to robot!")
                play_alarm()
            
            # Step 4: Visualization.
            for box_3d in boxes_3d:
                try:
                    class_name = box_3d['class_name'].lower()
                    bbox_coords = box_3d['bbox_2d']
                    
                    # If detection is for a robot, draw a semi-transparent blue overlay.
                    if 'robot' in class_name:
                        x1, y1, x2, y2 = map(int, bbox_coords)
                        overlay = result_frame.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
                        alpha = 0.5  # 50% transparency
                        result_frame = cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0)
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    else:
                        # For other detections, draw a standard rectangle.
                        x1, y1, x2, y2 = map(int, bbox_coords)
                        if 'person' in class_name:
                            color = (0, 255, 0)  # Green
                        else:
                            color = (255, 255, 255)  # White
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(result_frame, box_3d['class_name'], (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception as e:
                    print(f"Error drawing box: {e}")
                    continue
            
            if enable_bev:
                try:
                    bev.reset()
                    for box_3d in boxes_3d:
                        bev.draw_box(box_3d)
                    bev_image = bev.get_image()
                    bev_height = height // 4
                    bev_width = bev_height
                    if bev_height > 0 and bev_width > 0:
                        bev_resized = cv2.resize(bev_image, (bev_width, bev_height))
                        result_frame[height - bev_height:height, 0:bev_width] = bev_resized
                        cv2.rectangle(result_frame, (0, height - bev_height), (bev_width, height), (255, 255, 255), 1)
                        cv2.putText(result_frame, "Bird's Eye View", (10, height - bev_height + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    print(f"Error drawing BEV: {e}")
            
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed_time = time.time() - start_time
                fps_value = frame_count / elapsed_time
                fps_display = f"FPS: {fps_value:.1f}"
            cv2.putText(result_frame, f"{fps_display} | Device: {device}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            try:
                depth_height = height // 4
                depth_width = depth_height * width // height
                depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))
                result_frame[0:depth_height, 0:depth_width] = depth_resized
            except Exception as e:
                print(f"Error adding depth map: {e}")
            
            out.write(result_frame)
            
            cv2.imshow("3D Object Detection", result_frame)
            cv2.imshow("Depth Map", depth_colored)
            cv2.imshow("Object Detection", detection_frame)
            
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                print("Exiting program...")
                break
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                print("Exiting program...")
                break
            continue
    
    print("Cleaning up resources...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        cv2.destroyAllWindows()
