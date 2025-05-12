import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import pytz
import json
import os
from no_access_detection import load_no_access_events, get_available_dates

# Define Indian time zone
IST = pytz.timezone('Asia/Kolkata')

# Function to get current time in IST
def get_current_time_ist():
    return datetime.now(pytz.UTC).astimezone(IST)

# Function to create a table of no-access events for a specific date
def create_no_access_table(camera_id, selected_date=None):
    try:
        # Load no-access events
        events = load_no_access_events()
        
        if camera_id not in events or not events[camera_id]:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(events[camera_id])
        
        # Ensure timestamp is a datetime object
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Drop any rows where conversion failed
        df = df.dropna(subset=['timestamp'])
        
        if df.empty:
            return None
        
        # Apply timezone info correctly - ensure all timestamps are in IST
        df['timestamp'] = df['timestamp'].apply(
            lambda x: x if x.tzinfo else pd.Timestamp(x).tz_localize(pytz.UTC)
        ).dt.tz_convert(IST)
        
        # Filter by date if specified
        if selected_date:
            # Get start and end of the selected date in IST
            start_date = datetime.combine(selected_date, datetime.min.time()).replace(tzinfo=IST)
            end_date = datetime.combine(selected_date, datetime.max.time()).replace(tzinfo=IST)
            
            # Filter dataframe
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            
            if df.empty:
                return None
        else:
            # Last 24 hours only if no date specified
            last_24h = get_current_time_ist() - timedelta(hours=24)
            df = df[df['timestamp'] >= last_24h]
        
        if df.empty:
            return None
        
        # Format timestamp for display
        df['time'] = df['timestamp'].dt.strftime('%H:%M:%S')
        df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        
        # Create a clean table for display
        display_df = df[['date', 'time']].copy()
        display_df['event'] = 'Person Detected'
        
        # Rename columns for display
        display_df.columns = ['Date', 'Time', 'Event']
        
        return display_df
    
    except Exception as e:
        st.error(f"Error creating no-access table: {str(e)}")
        return None

# Function to display no-access events
def display_no_access_events(camera_id, selected_date=None):
    # Get camera name
    camera_name = "Unknown Camera"
    if 'cameras' in st.session_state:
        for cam in st.session_state.cameras:
            if cam.get('camera_id') == camera_id:
                camera_name = cam['name']
                break
    
    # Create title
    date_str = selected_date.strftime('%Y-%m-%d') if selected_date else "Last 24 Hours"
    st.subheader(f"No-Access Events - {camera_name} on {date_str}")
    
    # Get and display table
    table_df = create_no_access_table(camera_id, selected_date)
    
    if table_df is not None:
        st.dataframe(table_df, use_container_width=True)
        st.write(f"Total events: {len(table_df)}")
    else:
        st.info("No no-access events recorded for this camera on the selected date.")
