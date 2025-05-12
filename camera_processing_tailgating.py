import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime, timedelta
import pytz
from tailgating_detection import save_tailgating_event

# Define Indian time zone
IST = pytz.timezone('Asia/Kolkata')

# Function to get current time in IST
def get_current_time_ist():
    return datetime.now(pytz.UTC).astimezone(IST)

# Function to process tailgating detection
def process_tailgating_detection(i, camera, frame_data):
    camera_id = camera['camera_id']
    current_time = get_current_time_ist()
    
    # Initialize cooldown entry if it doesn't exist
    if camera_id not in st.session_state.tailgating_cooldowns:
        st.session_state.tailgating_cooldowns[camera_id] = {
            'in_cooldown': False,
            'cooldown_until': current_time
        }
    
    cooldown_info = st.session_state.tailgating_cooldowns[camera_id]
    
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
            f"TAILGATING COOLDOWN: {remaining_minutes}m {remaining_seconds}s", 
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
            
            # Save tailgating event if multiple people detected
            if count > 1:
                timestamp = get_current_time_ist().isoformat()
                save_tailgating_event(camera['camera_id'], timestamp, count)
                
                # Start cooldown period (2 minutes)
                cooldown_info['in_cooldown'] = True
                cooldown_info['cooldown_until'] = current_time + timedelta(minutes=2)
                cooldown_info['last_processed_frame'] = processed_frame
                
                # Add text about cooldown starting
                img = Image.open(io.BytesIO(processed_frame))
                img_array = np.array(img)
                cv2.putText(
                    img_array, 
                    f"TAILGATING DETECTED: {count} people - COOLDOWN: 2m 0s", 
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
            st.error(f"Error in tailgating detection: {str(e)}")
            st.session_state.cameras[i]["last_frame"] = frame_data

# Function to optimize frame (reduce size and quality)
def optimize_frame(i, camera, frame_data):
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Resize if needed
        if st.session_state.resize_factor < 1.0:
            h, w = img.shape[:2]
            new_h, new_w = int(h * st.session_state.resize_factor), int(w * st.session_state.resize_factor)
            img = cv2.resize(img, (new_w, new_h))
        
        # Compress with specified quality
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), st.session_state.image_quality]
        _, buffer = cv2.imencode('.jpg', img, encode_params)
        st.session_state.cameras[i]["last_frame"] = buffer.tobytes()
    except:
        # If any error occurs, use the original frame
        st.session_state.cameras[i]["last_frame"] = frame_data
