import json
import os
import random
from datetime import datetime, timedelta
import pytz
import uuid

# Define Indian time zone
IST = pytz.timezone('Asia/Kolkata')

# Function to get current time in IST
def get_current_time_ist():
    return datetime.now(pytz.UTC).astimezone(IST)

# Function to load cameras from JSON file
def load_cameras():
    try:
        if os.path.exists('cameras.json'):
            with open('cameras.json', 'r') as file:
                cameras = json.load(file)
                return cameras
        else:
            # Create a sample camera if none exists
            sample_camera = [{
                'name': 'Sample Camera',
                'url': 'http://example.com/camera1',
                'camera_id': str(uuid.uuid4()),
                'detection_active': True
            }]
            with open('cameras.json', 'w') as file:
                json.dump(sample_camera, file, indent=4)
            return sample_camera
    except Exception as e:
        print(f"Error loading cameras: {str(e)}")
        return []

# Function to load occupancy history
def load_occupancy_history():
    try:
        if os.path.exists('occupancy_history.json'):
            with open('occupancy_history.json', 'r') as file:
                history = json.load(file)
                return history
        else:
            return {}
    except Exception as e:
        print(f"Error loading occupancy history: {str(e)}")
        return {}

# Function to save occupancy data
def save_occupancy_history(history):
    try:
        with open('occupancy_history.json', 'w') as file:
            json.dump(history, file, indent=4)
        return True
    except Exception as e:
        print(f"Error saving occupancy data: {str(e)}")
        return False

# Generate sample occupancy data for the past 7 days
def generate_sample_data():
    # Load cameras
    cameras = load_cameras()
    if not cameras:
        print("No cameras found. Creating a sample camera.")
        cameras = [{
            'name': 'Sample Camera',
            'url': 'http://example.com/camera1',
            'camera_id': str(uuid.uuid4()),
            'detection_active': True
        }]
        with open('cameras.json', 'w') as file:
            json.dump(cameras, file, indent=4)
    
    # Load existing history or create new
    history = load_occupancy_history()
    
    # Current time in IST
    current_time = get_current_time_ist()
    
    # Generate data for each camera
    for camera in cameras:
        camera_id = camera['camera_id']
        
        # Initialize camera entry if it doesn't exist
        if camera_id not in history:
            history[camera_id] = []
        
        # Generate data for past 7 days
        for day in range(7):
            # Start from 7 days ago
            date = current_time - timedelta(days=day)
            date_start = datetime.combine(date.date(), datetime.min.time()).replace(tzinfo=IST)
            
            # Generate data for each hour of the day
            for hour in range(24):
                # Morning hours (8-11): High occupancy
                if 8 <= hour <= 11:
                    max_occupancy = random.randint(5, 15)
                # Lunch hours (12-14): Medium occupancy
                elif 12 <= hour <= 14:
                    max_occupancy = random.randint(3, 8)
                # Afternoon hours (15-18): High occupancy
                elif 15 <= hour <= 18:
                    max_occupancy = random.randint(6, 12)
                # Evening hours (19-22): Medium-low occupancy
                elif 19 <= hour <= 22:
                    max_occupancy = random.randint(2, 7)
                # Night hours: Low or no occupancy
                else:
                    max_occupancy = random.randint(0, 3)
                
                # Generate random number of data points for this hour
                num_points = random.randint(3, 10)
                
                for _ in range(num_points):
                    # Random minute within the hour
                    minute = random.randint(0, 59)
                    timestamp = date_start + timedelta(hours=hour, minutes=minute)
                    
                    # Random occupancy count for this timestamp (up to the max for this hour)
                    count = random.randint(0, max_occupancy)
                    
                    # Add data point
                    history[camera_id].append({
                        'timestamp': timestamp.isoformat(),
                        'count': count
                    })
    
    # Save the updated history
    success = save_occupancy_history(history)
    if success:
        print("Sample data generated successfully!")
        # Count total data points
        total_points = sum(len(history[camera_id]) for camera_id in history)
        print(f"Generated {total_points} data points across {len(cameras)} cameras for the past 7 days.")
    else:
        print("Failed to save sample data.")

if __name__ == "__main__":
    generate_sample_data()
