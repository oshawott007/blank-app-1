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

# Function to load no-access events
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

# Function to save no-access events
def save_no_access_events(events):
    try:
        with open('no_access_events.json', 'w') as file:
            json.dump(events, file, indent=4)
        return True
    except Exception as e:
        print(f"Error saving no-access events: {str(e)}")
        return False

# Generate sample no-access events for the past 7 days
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
    
    # Load existing events or create new
    events = load_no_access_events()
    
    # Current time in IST
    current_time = get_current_time_ist()
    
    # Generate data for each camera
    for camera in cameras:
        camera_id = camera['camera_id']
        
        # Initialize camera entry if it doesn't exist
        if camera_id not in events:
            events[camera_id] = []
        
        # Generate data for past 7 days
        for day in range(7):
            # Start from 7 days ago
            date = current_time - timedelta(days=day)
            date_start = datetime.combine(date.date(), datetime.min.time()).replace(tzinfo=IST)
            
            # Generate random number of events for this day (more events during night hours)
            num_events = random.randint(5, 15)
            
            for _ in range(num_events):
                # Generate random hour - more likely during night hours (19-6)
                if random.random() < 0.7:  # 70% chance of night events
                    hour = random.choice(list(range(19, 24)) + list(range(0, 7)))
                else:
                    hour = random.randint(7, 18)
                
                # Random minute
                minute = random.randint(0, 59)
                # Random second
                second = random.randint(0, 59)
                
                # Create timestamp
                timestamp = date_start + timedelta(hours=hour, minutes=minute, seconds=second)
                
                # Add event
                events[camera_id].append({
                    'timestamp': timestamp.isoformat(),
                    'image_path': None  # No actual image path for sample data
                })
    
    # Save the updated events
    success = save_no_access_events(events)
    if success:
        print("Sample no-access events generated successfully!")
        # Count total events
        total_events = sum(len(events[camera_id]) for camera_id in events)
        print(f"Generated {total_events} no-access events across {len(cameras)} cameras for the past 7 days.")
    else:
        print("Failed to save sample no-access events.")

if __name__ == "__main__":
    generate_sample_data()
