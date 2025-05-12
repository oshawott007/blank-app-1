import json
import os
from datetime import datetime
import pytz

# Define Indian time zone
IST = pytz.timezone('Asia/Kolkata')

# Function to get current time in IST
def get_current_time_ist():
    return datetime.now(pytz.UTC).astimezone(IST)

# Function to load tailgating events from JSON file
def load_tailgating_events():
    try:
        if os.path.exists('tailgating_events.json'):
            with open('tailgating_events.json', 'r') as file:
                events = json.load(file)
                return events
        else:
            return {}
    except Exception as e:
        print(f"Error loading tailgating events: {str(e)}")
        return {}

# Function to save tailgating event
def save_tailgating_event(camera_id, timestamp, person_count):
    try:
        # Load existing data
        events = load_tailgating_events()
        
        # Initialize camera entry if it doesn't exist
        if camera_id not in events:
            events[camera_id] = []
        
        # Add new entry
        events[camera_id].append({
            'timestamp': timestamp,
            'person_count': person_count
        })
        
        # Save updated data
        with open('tailgating_events.json', 'w') as file:
            json.dump(events, file, indent=4)
        return True
    except Exception as e:
        print(f"Error saving tailgating event: {str(e)}")
        return False

# Function to get available dates for tailgating events
def get_available_dates(camera_id):
    try:
        events = load_tailgating_events()
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
