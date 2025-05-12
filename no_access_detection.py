import cv2
import numpy as np
import json
import os
from datetime import datetime
import pytz
import onnxruntime as ort

# Define Indian time zone
IST = pytz.timezone('Asia/Kolkata')

# Function to get current time in IST
def get_current_time_ist():
    return datetime.now(pytz.UTC).astimezone(IST)

# Function to load no-access events from JSON file
def load_no_access_events():
    try:
        if os.path.exists('no_access_events.json'):
            with open('no_access_events.json', 'r') as file:
                events = json.load(file)
                return events
        else:
            return {}
    except Exception as e:
        print(f"Error loading no-access events: {str(e)}")
        return {}

# Function to save no-access event
def save_no_access_event(camera_id, timestamp, image_path=None):
    try:
        # Load existing data
        events = load_no_access_events()

        # Initialize camera entry if it doesn't exist
        if camera_id not in events:
            events[camera_id] = []

        # Add new entry
        events[camera_id].append({
            'timestamp': timestamp,
            'image_path': image_path
        })

        # Save updated data
        with open('no_access_events.json', 'w') as file:
            json.dump(events, file, indent=4)
        return True
    except Exception as e:
        print(f"Error saving no-access event: {str(e)}")
        return False

# Function to get available dates for no-access events
def get_available_dates(camera_id):
    try:
        events = load_no_access_events()
        if camera_id not in events or not events[camera_id]:
            return []

        # Convert timestamps to datetime objects
        dates = []
        for event in events[camera_id]:
            try:
                dt = datetime.fromisoformat(event['timestamp'])
                if not dt.tzinfo:
                    dt = dt.replace(tzinfo=pytz.UTC).astimezone(IST)
                dates.append(dt.date())
            except (ValueError, TypeError):
                continue

        # Get unique dates
        unique_dates = list(set(dates))

        return sorted(unique_dates)
    except Exception as e:
        print(f"Error getting available dates: {str(e)}")
        return []

# YOLOv8 ONNX detection
class YOLOv8Detector:
    def __init__(self, model_path='yolov8n.onnx'):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # COCO class names (only need person which is class 0)
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                           'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                           'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                           'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                           'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                           'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def detect(self, image_bytes):
        try:
            import streamlit as st

            # Get performance settings from session state
            image_quality = st.session_state.image_quality
            resize_factor = st.session_state.resize_factor

            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Resize image for faster processing if needed
            if resize_factor < 1.0:
                h, w = img.shape[:2]
                new_h, new_w = int(h * resize_factor), int(w * resize_factor)
                img_resized = cv2.resize(img, (new_w, new_h))
            else:
                img_resized = img

            # Prepare image for inference
            input_img = self._prepare_input(img_resized)

            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_img})

            # Process results
            boxes, scores, class_ids = self._process_output(outputs)

            # If we resized the image, scale the boxes back to original size
            if resize_factor < 1.0:
                scale_factor = 1.0 / resize_factor
                scaled_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    x1 = x1 * scale_factor
                    y1 = y1 * scale_factor
                    x2 = x2 * scale_factor
                    y2 = y2 * scale_factor
                    scaled_boxes.append([x1, y1, x2, y2])
                boxes = scaled_boxes

            # Draw bounding boxes on original image
            result_img = self._draw_detections(img, boxes, scores, class_ids)

            # Count people
            person_count = sum(1 for class_id in class_ids if class_id == 0)  # 0 is the class ID for 'person'

            # Convert back to JPEG with specified quality
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), image_quality]
            _, buffer = cv2.imencode('.jpg', result_img, encode_params)
            jpg_bytes = buffer.tobytes()

            return person_count, jpg_bytes
        except Exception as e:
            print(f"Detection error: {str(e)}")
            return 0, image_bytes

    def _prepare_input(self, img):
        # Resize and normalize image
        input_img = cv2.resize(img, (640, 640))
        input_img = input_img.transpose(2, 0, 1)  # HWC to CHW
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, axis=0)
        return input_img

    def _process_output(self, outputs):
        # Process YOLOv8 output
        predictions = outputs[0]

        # Filter by confidence and class (only people)
        boxes = []
        scores = []
        class_ids = []

        for prediction in predictions:
            class_scores = prediction[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence > 0.5 and class_id == 0:  # 0.5 confidence threshold and class 0 is person
                x, y, w, h = prediction[0:4]

                # Convert to corner format
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2

                boxes.append([x1, y1, x2, y2])
                scores.append(float(confidence))
                class_ids.append(class_id)

        return boxes, scores, class_ids

    def _draw_detections(self, img, boxes, scores, class_ids):
        result_img = img.copy()

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box

            # Scale to image size
            h, w = img.shape[:2]
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)

            # Draw bounding box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Draw label
            label = f"{self.class_names[class_id]}: {score:.2f}"
            cv2.putText(result_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Add count text
        person_count = len(boxes)
        cv2.putText(result_img, f'NO ACCESS: {person_count} person(s) detected', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return result_img
