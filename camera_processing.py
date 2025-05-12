import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import requests
from datetime import datetime, timedelta
import pytz

# Import detection functions
from no_access_detection import save_no_access_event
from tailgating_detection import save_tailgating_event

# Define Indian time zone
IST = pytz.timezone('Asia/Kolkata')

# Function to get current time in IST
def get_current_time_ist():
    return datetime.now(pytz.UTC).astimezone(IST)

# Function to get a single frame from MJPEG stream
def get_mjpeg_frame(url, timeout=3):
    try:
        # Use a session for connection pooling
        session = requests.Session()

        # Set headers to optimize connection
        headers = {
            'Connection': 'keep-alive',
            'Accept-Encoding': 'gzip, deflate',
            'Accept': 'image/jpeg',
            'User-Agent': 'Mozilla/5.0'
        }

        # Make request with short timeout
        response = session.get(url, stream=True, timeout=timeout, headers=headers)

        if response.status_code == 200:
            # Read bytes more efficiently
            bytes_data = bytearray()
            content_length = 0
            max_size = 300000  # ~300KB max to prevent memory issues

            # Use a more efficient approach for finding JPEG frames
            for chunk in response.iter_content(chunk_size=4096):  # Larger chunks for efficiency
                bytes_data.extend(chunk)
                content_length += len(chunk)

                # Find JPEG frame boundaries
                a = bytes_data.find(b'\xff\xd8')  # JPEG start
                if a != -1:
                    # Once we find the start, look for the end
                    b = bytes_data.find(b'\xff\xd9', a)  # JPEG end
                    if b != -1:
                        # Extract the JPEG frame
                        jpg = bytes(bytes_data[a:b+2])

                        # Close the connection to free resources
                        response.close()
                        session.close()

                        return jpg, None

                # Prevent too much data accumulation
                if content_length > max_size:
                    response.close()
                    session.close()
                    break

            # Clean up resources
            response.close()
            session.close()
            return None, "Could not find complete JPEG frame"
        else:
            return None, f"HTTP error: {response.status_code}"
    except requests.exceptions.Timeout:
        return None, "Connection timeout"
    except requests.exceptions.ConnectionError:
        return None, "Connection error"
    except requests.exceptions.RequestException as e:
        return None, f"Request error: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"
    finally:
        # Ensure resources are cleaned up
        try:
            if 'response' in locals() and response:
                response.close()
            if 'session' in locals() and session:
                session.close()
        except:
            pass

# Human detection using OpenCV HOG detector
def detect_humans(image_bytes):
    try:
        # Get image quality from session state
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

        # Initialize HOG detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Detect people with optimized parameters
        # Use larger winStride for faster detection
        # Use smaller scale for better performance
        boxes, _ = hog.detectMultiScale(
            img_resized,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05  # Smaller scale factor for faster detection
        )

        # If we resized the image, scale the boxes back to original size
        if resize_factor < 1.0:
            scale_factor = 1.0 / resize_factor
            scaled_boxes = []
            for (x, y, w, h) in boxes:
                x = int(x * scale_factor)
                y = int(y * scale_factor)
                w = int(w * scale_factor)
                h = int(h * scale_factor)
                scaled_boxes.append((x, y, w, h))
            boxes = scaled_boxes

        # Count people
        person_count = len(boxes)

        # Draw bounding boxes if people detected
        if person_count > 0:
            for (x, y, w, h) in boxes:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Add count text
            cv2.putText(img, f'People: {person_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Convert back to JPEG with specified quality
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), image_quality]
        _, buffer = cv2.imencode('.jpg', img, encode_params)
        jpg_bytes = buffer.tobytes()

        return person_count, jpg_bytes
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return 0, image_bytes

# Function to process camera frames
def process_camera_frames(cameras, save_occupancy_data):
    # Import required functions
    from camera_processing_tailgating import process_tailgating_detection, optimize_frame

    # Initialize last detection time if not set
    if 'last_detection' not in st.session_state:
        st.session_state.last_detection = get_current_time_ist() - timedelta(seconds=st.session_state.detection_interval)
        st.session_state.frame_counter = 0

    # Increment frame counter
    if 'frame_counter' not in st.session_state:
        st.session_state.frame_counter = 0
    st.session_state.frame_counter += 1

    # Check if it's time to run detection based on frame skip and detection interval
    current_time = get_current_time_ist()
    detection_time_diff = (current_time - st.session_state.last_detection).total_seconds()
    run_detection = (st.session_state.frame_counter % st.session_state.frame_skip == 0) and (detection_time_diff >= st.session_state.detection_interval)

    # If it's time to run detection, update the last detection time
    if run_detection:
        st.session_state.last_detection = current_time

    # Process each camera
    for i, camera in enumerate(cameras):
        # Get frame from camera
        frame_data, error = get_mjpeg_frame(camera['url'])

        if frame_data:
            # Get detection modes
            occupancy_active = camera.get('detection_active', False)
            no_access_active = camera.get('no_access_active', False)
            tailgating_active = camera.get('tailgating_active', False)

            # Process frame based on detection mode and whether it's time to run detection
            if run_detection:
                # Occupancy detection
                if occupancy_active:
                    process_occupancy_detection(i, camera, frame_data, save_occupancy_data)

                # No-access detection
                elif no_access_active and 'yolo_detector' in st.session_state and st.session_state.yolo_detector:
                    process_no_access_detection(i, camera, frame_data)

                # Tailgating detection
                elif tailgating_active and 'yolo_detector' in st.session_state and st.session_state.yolo_detector:
                    process_tailgating_detection(i, camera, frame_data)

                # No detection active, just optimize the frame
                else:
                    optimize_frame(i, camera, frame_data)

            # If not running detection, just optimize the frame
            else:
                optimize_frame(i, camera, frame_data)

            # Update camera status
            cameras[i]["status"] = "Connected"
        else:
            # Update camera status with error
            cameras[i]["status"] = f"Error: {error}"

    return cameras

# Function to process occupancy detection
def process_occupancy_detection(i, camera, frame_data, save_occupancy_data):

    # Detect humans in frame
    count, processed_frame = detect_humans(frame_data)
    st.session_state.cameras[i]["last_frame"] = processed_frame
    st.session_state.cameras[i]["person_count"] = count

    # Save occupancy data with IST timestamp
    timestamp = get_current_time_ist().isoformat()
    save_occupancy_data(camera['camera_id'], timestamp, count)

# Function to process no-access detection
def process_no_access_detection(i, camera, frame_data):
    camera_id = camera['camera_id']
    current_time = get_current_time_ist()

    # Initialize cooldown entry if it doesn't exist
    if camera_id not in st.session_state.no_access_cooldowns:
        st.session_state.no_access_cooldowns[camera_id] = {
            'in_cooldown': False,
            'cooldown_until': current_time
        }

    cooldown_info = st.session_state.no_access_cooldowns[camera_id]

    # Check if we're in cooldown period
    if cooldown_info['in_cooldown'] and current_time < cooldown_info['cooldown_until']:
        # Still in cooldown, display the last processed frame with cooldown message
        if 'last_processed_frame' in cooldown_info:
            st.session_state.cameras[i]["last_frame"] = cooldown_info['last_processed_frame']
        else:
            st.session_state.cameras[i]["last_frame"] = frame_data

        # Calculate remaining cooldown time
        remaining_seconds = (cooldown_info['cooldown_until'] - current_time).total_seconds()
        remaining_minutes = int(remaining_seconds // 60)
        remaining_seconds = int(remaining_seconds % 60)

        # Add cooldown message to the frame
        img = Image.open(io.BytesIO(st.session_state.cameras[i]["last_frame"]))
        img_array = np.array(img)

        # Add text with cooldown information
        cv2.putText(
            img_array,
            f"NO-ACCESS COOLDOWN: {remaining_minutes}m {remaining_seconds}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # Convert back to bytes
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), st.session_state.image_quality]
        _, buffer = cv2.imencode('.jpg', img_array, encode_params)
        st.session_state.cameras[i]["last_frame"] = buffer.tobytes()
    else:
        # Not in cooldown or cooldown expired, perform detection
        try:
            count, processed_frame = st.session_state.yolo_detector.detect(frame_data)
            st.session_state.cameras[i]["last_frame"] = processed_frame
            st.session_state.cameras[i]["person_count"] = count

            # Save no-access event if people detected
            if count > 0:
                timestamp = get_current_time_ist().isoformat()
                save_no_access_event(camera['camera_id'], timestamp)

                # Start cooldown period (5 minutes)
                cooldown_info['in_cooldown'] = True
                cooldown_info['cooldown_until'] = current_time + timedelta(minutes=5)
                cooldown_info['last_processed_frame'] = processed_frame

                # Add text about cooldown starting
                img = Image.open(io.BytesIO(processed_frame))
                img_array = np.array(img)
                cv2.putText(
                    img_array,
                    "NO-ACCESS COOLDOWN STARTED: 5m 0s",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                # Convert back to bytes
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), st.session_state.image_quality]
                _, buffer = cv2.imencode('.jpg', img_array, encode_params)
                st.session_state.cameras[i]["last_frame"] = buffer.tobytes()
        except Exception as e:
            st.error(f"Error in no-access detection: {str(e)}")
            st.session_state.cameras[i]["last_frame"] = frame_data
