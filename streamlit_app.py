

import streamlit as st
import requests
from PIL import Image
import io
import time
from datetime import datetime, timedelta, date
import pytz
import base64
import json
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import threading
from collections import defaultdict
import uuid

# Import no-access detection functionality
from no_access_detection import YOLOv8Detector, save_no_access_event, get_current_time_ist as get_ist_time
from no_access_view import display_no_access_events

# Import tailgating detection functionality
from tailgating_detection import save_tailgating_event
from tailgating_view import display_tailgating_events

# Import camera processing modules
from camera_processing import process_camera_frames, process_occupancy_detection, process_no_access_detection, get_mjpeg_frame, detect_humans
from camera_processing_tailgating import process_tailgating_detection, optimize_frame

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
                # Initialize runtime fields for each camera
                for camera in cameras:
                    if 'last_frame' not in camera:
                        camera['last_frame'] = None
                    if 'status' not in camera:
                        camera['status'] = "Connecting..."
                    if 'detection_active' not in camera:
                        camera['detection_active'] = False
                    if 'camera_id' not in camera:
                        camera['camera_id'] = str(uuid.uuid4())
                return cameras
        else:
            return []
    except Exception as e:
        st.error(f"Error loading cameras: {str(e)}")
        return []

# Function to save cameras to JSON file
def save_cameras(cameras):
    try:
        # Create a copy without runtime data
        cameras_to_save = []
        for camera in cameras:
            cameras_to_save.append({
                'name': camera['name'],
                'url': camera['url'],
                'camera_id': camera.get('camera_id', str(uuid.uuid4())),
                'detection_active': camera.get('detection_active', False)
            })

        with open('cameras.json', 'w') as file:
            json.dump(cameras_to_save, file, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving cameras: {str(e)}")
        return False

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
        st.error(f"Error loading occupancy history: {str(e)}")
        return {}

# Function to save occupancy data
def save_occupancy_data(camera_id, timestamp, count):
    try:
        # Load existing data
        history = load_occupancy_history()

        # Initialize camera entry if it doesn't exist
        if camera_id not in history:
            history[camera_id] = []

        # Add new entry
        history[camera_id].append({
            'timestamp': timestamp,
            'count': count
        })

        # Save updated data
        with open('occupancy_history.json', 'w') as file:
            json.dump(history, file, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving occupancy data: {str(e)}")
        return False

# These functions are now imported from camera_processing.py

# Function to create hourly occupancy line graph for a specific date
def create_hourly_graph(camera_id, selected_date=None):
    try:
        # Load occupancy data
        history = load_occupancy_history()

        if camera_id not in history or not history[camera_id]:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(history[camera_id])

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

        # Create hour bins for full 24 hours (0-23)
        hours = list(range(24))
        hourly_max = pd.DataFrame(hours, columns=['hour'])
        hourly_max['count'] = 0
        # Sort by hour to ensure proper display order
        hourly_max = hourly_max.sort_values('hour')

        # Extract hour from timestamp and group by hour
        df['hour'] = df['timestamp'].dt.hour
        hour_counts = df.groupby('hour')['count'].max().reset_index()

        # Update the hourly_max dataframe with actual counts
        for _, row in hour_counts.iterrows():  # Use _ for unused index variable
            hourly_max.loc[hourly_max['hour'] == row['hour'], 'count'] = row['count']

        # Get camera name for title
        camera_name = "Unknown Camera"
        for cam in st.session_state.cameras:
            if cam.get('camera_id') == camera_id:
                camera_name = cam['name']
                break

        # Create graph
        date_str = selected_date.strftime('%Y-%m-%d') if selected_date else "Last 24 Hours"
        title = f"Hourly Maximum Occupancy - {camera_name} on {date_str}"

        # Format x-axis to show hours in 24-hour format with IST indicator
        hourly_max['hour_str'] = hourly_max['hour'].apply(lambda x: f"{x:02d}:00")
        # Ensure hours are in correct order
        hourly_max = hourly_max.sort_values('hour')

        fig = px.line(
            hourly_max,
            x='hour_str',
            y='count',
            title=title,
            labels={'hour_str': 'Hour of Day', 'count': 'Maximum People Count'}
        )

        # Customize layout to match the style in the image
        fig.update_layout(
            xaxis=dict(
                title='Hour of Day',
                gridcolor='lightgray',
                showgrid=True,
                tickmode='array',
                tickvals=hourly_max['hour_str'].tolist(),
                ticktext=hourly_max['hour_str'].tolist(),
                categoryorder='array',
                categoryarray=hourly_max['hour_str'].tolist(),
                color='black',  # Make axis text black
                tickangle=45,   # Angle the tick labels for better readability
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title='Maximum People Count',
                gridcolor='lightgray',
                showgrid=True,
                color='black',  # Make axis text black
                tickfont=dict(size=10)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, color='black'),  # Make all text black
            margin=dict(l=40, r=40, t=50, b=40),
            hovermode='x unified',
            title=dict(
                font=dict(color='black', size=14)  # Make title text black
            ),
            height=400,  # Taller for better visibility
            width=700    # Wider for vertical layout
        )

        # Set line color to blue
        fig.update_traces(line=dict(color='blue', width=2))

        # Make axis text darker
        fig.update_xaxes(tickfont=dict(color='black', size=12))
        fig.update_yaxes(tickfont=dict(color='black', size=12))

        return fig
    except Exception as e:
        st.error(f"Error creating hourly graph: {str(e)}")
        return None

# Function to create circular occupancy graph (24-hour clock) for a specific date
def create_circular_graph(camera_id, selected_date=None):
    try:
        # Load occupancy data
        history = load_occupancy_history()

        if camera_id not in history or not history[camera_id]:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(history[camera_id])

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

        # Create binary occupancy (0 or 1)
        df['occupied'] = df['count'].apply(lambda x: 1 if x > 0 else 0)

        # Extract hours and minutes for polar coordinates
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute

        # Convert to angle (0 to 2π) - adjust to make it like a clock (12 at top, 3 at right, etc.)
        # For a clock: 0 hour is at the top (270 degrees or -90 degrees in standard position)
        df['angle'] = 2 * np.pi * (df['hour'] + df['minute']/60) / 24
        # Rotate to make it like a clock (subtract 90 degrees or π/2 radians)
        df['angle'] = df['angle'] - (np.pi/2)

        # Convert to cartesian coordinates
        df['x'] = np.cos(df['angle'])
        df['y'] = np.sin(df['angle'])

        # Create scatter plot with polar coordinates
        fig = go.Figure()

        # Add minute markers (thin lines from center) - match the style in the image
        for _, row in df.iterrows():
            if row['occupied'] == 1:
                color = 'orange'  # Use orange color for occupied points as in the image
            else:
                continue  # Skip unoccupied points

            fig.add_trace(go.Scatter(
                x=[0, row['x']],
                y=[0, row['y']],
                mode='lines',
                line=dict(color=color, width=1),
                hoverinfo='text',
                hovertext=f"Time: {row['hour']:02d}:{row['minute']:02d}, Count: {row['count']}",
                showlegend=False
            ))

        # Add circle for reference
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none',
            showlegend=False
        ))

        # Add hour markers - adjust to make it like a clock (12 at top, 3 at right, etc.)
        for hour in range(24):
            # Calculate angle and adjust for clock orientation (subtract 90 degrees or π/2 radians)
            angle = (2 * np.pi * hour / 24) - (np.pi/2)
            x = 1.1 * np.cos(angle)
            y = 1.1 * np.sin(angle)

            # Add hour label in format matching the image
            fig.add_annotation(
                x=x, y=y,
                text=f"{hour:02d}:00",
                showarrow=False,
                font=dict(size=10, color='black')
            )

            # Add tick mark
            x1, y1 = np.cos(angle), np.sin(angle)
            x2, y2 = 1.05 * np.cos(angle), 1.05 * np.sin(angle)
            fig.add_shape(
                type="line",
                x0=x1, y0=y1,
                x1=x2, y1=y2,
                line=dict(color="black", width=1.5),  # Make tick marks darker
            )

        # Get camera name for title
        camera_name = "Unknown Camera"
        for cam in st.session_state.cameras:
            if cam.get('camera_id') == camera_id:
                camera_name = cam['name']
                break

        # Add title
        date_str = selected_date.strftime('%Y-%m-%d') if selected_date else "Last 24 Hours"
        title = f"Minute-by-Minute Presence - {camera_name} on {date_str}"

        # Update layout to match the style in the image
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(color='black', size=14)  # Make title text black
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-1.2, 1.2]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-1.2, 1.2],
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20),
            width=700,  # Wider for vertical layout
            height=400,  # Taller for better visibility
            showlegend=False,
            font=dict(color='black')  # Make all text black
        )

        return fig
    except Exception as e:
        st.error(f"Error creating circular graph: {str(e)}")
        return None

# Get available dates for a camera
def get_available_dates(camera_id):
    try:
        history = load_occupancy_history()
        if camera_id not in history or not history[camera_id]:
            return []

        df = pd.DataFrame(history[camera_id])

        # Ensure timestamp is a datetime object with error handling
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Drop any rows where conversion failed
        df = df.dropna(subset=['timestamp'])

        if df.empty:
            return []

        # Apply timezone info correctly - ensure all timestamps are in IST
        df['timestamp'] = df['timestamp'].apply(
            lambda x: x if x.tzinfo else pd.Timestamp(x).tz_localize(pytz.UTC)
        ).dt.tz_convert(IST)

        # Extract the date component
        df['date'] = df['timestamp'].dt.date

        # Get unique dates
        unique_dates = df['date'].unique()

        return sorted(unique_dates)
    except Exception as e:
        st.error(f"Error getting available dates: {str(e)}")
        return []

# Initialize session state
if 'initialized' not in st.session_state:
    # Load cameras from JSON file
    st.session_state.cameras = load_cameras()
    st.session_state.last_refresh = get_current_time_ist()
    st.session_state.initialized = True
    st.session_state.save_success = None
    st.session_state.active_history_camera = None
    st.session_state.selected_date = None
    st.session_state.view_mode = "occupancy"  # Default view mode
    st.session_state.no_access_cooldowns = {}  # Store cooldown times for each camera
    st.session_state.tailgating_cooldowns = {}  # Store cooldown times for tailgating detection

    # Performance settings
    st.session_state.performance_mode = "balanced"  # Options: low, balanced, high
    st.session_state.frame_skip = 1  # Process every nth frame for detection
    st.session_state.detection_interval = 3  # Seconds between detection runs
    st.session_state.image_quality = 80  # JPEG quality (0-100)
    st.session_state.resize_factor = 1.0  # Resize factor for display (1.0 = original size)

    # Initialize YOLOv8 detector
    try:
        st.session_state.yolo_detector = YOLOv8Detector('yolov8n.onnx')
    except Exception as e:
        st.error(f"Error initializing YOLOv8 detector: {str(e)}")
        st.session_state.yolo_detector = None

# App title
st.title("IP Camera Viewer with Occupancy Detection")

# Set default performance settings (balanced mode)
st.session_state.frame_skip = 2
st.session_state.detection_interval = 3
st.session_state.image_quality = 80
st.session_state.resize_factor = 0.9

# Camera addition form
st.subheader("Add Camera")
with st.form("add_camera_form"):
    col1, col2 = st.columns(2)
    with col1:
        camera_name = st.text_input("Camera Name")
    with col2:
        camera_url = st.text_input("Camera URL")

    submit_button = st.form_submit_button("Add Camera")

    if submit_button and camera_name and camera_url:
        # Check if camera with same name already exists
        exists = False
        for cam in st.session_state.cameras:
            if cam["name"] == camera_name:
                exists = True
                break

        if not exists:
            new_camera = {
                "name": camera_name,
                "url": camera_url,
                "last_frame": None,
                "status": "Connecting...",
                "detection_active": False,
                "camera_id": str(uuid.uuid4())
            }
            st.session_state.cameras.append(new_camera)
            # Save to JSON file
            save_success = save_cameras(st.session_state.cameras)
            if save_success:
                st.session_state.save_success = "Camera added and saved successfully!"
            else:
                st.session_state.save_success = "Camera added but failed to save to file."
            st.rerun()
        else:
            st.error(f"Camera with name '{camera_name}' already exists!")

# Display save status if available
if st.session_state.save_success:
    st.success(st.session_state.save_success)
    st.session_state.save_success = None

# Camera table display
if st.session_state.cameras:
    st.subheader("Camera List")

    # Create header row for our custom table
    header_cols = st.columns([2, 4, 1])
    with header_cols[0]:
        st.markdown("**Camera Name**")
    with header_cols[1]:
        st.markdown("**Camera URL**")
    with header_cols[2]:
        st.markdown("**Action**")

    st.markdown("---")  # Divider

    # Create rows for each camera
    for i, camera in enumerate(st.session_state.cameras):
        cols = st.columns([2, 4, 1])
        with cols[0]:
            st.write(camera["name"])
        with cols[1]:
            st.write(camera["url"])
        with cols[2]:
            # Unique key for each button to avoid conflicts
            if st.button("Remove", key=f"remove_btn_{i}"):
                st.session_state.cameras.pop(i)
                # Save changes to JSON file
                save_success = save_cameras(st.session_state.cameras)
                if save_success:
                    st.session_state.save_success = "Camera removed and changes saved successfully!"
                else:
                    st.session_state.save_success = "Camera removed but failed to save changes."
                st.rerun()

# Refresh control - automatic refresh every 10 seconds
refresh_rate = 1  # Base refresh rate for UI updates (1 second)
auto_refresh_rate = 10  # Auto refresh every 10 seconds

# Add auto-refresh information
current_time_ist = get_current_time_ist()
st.info(f"Cameras auto-refresh every 10 seconds. Last updated: {current_time_ist.strftime('%H:%M:%S')} IST")

# Set up auto-refresh
if 'last_auto_refresh' not in st.session_state:
    st.session_state.last_auto_refresh = get_current_time_ist()

# Check if it's time for auto-refresh (only if not in history view)
auto_refresh_time_diff = (current_time_ist - st.session_state.last_auto_refresh).total_seconds()
if auto_refresh_time_diff >= auto_refresh_rate and not st.session_state.active_history_camera:
    st.session_state.last_auto_refresh = current_time_ist
    st.session_state.last_refresh = current_time_ist
    st.session_state.last_detection = current_time_ist - timedelta(seconds=st.session_state.detection_interval)
    # Use rerun to force a complete refresh
    st.rerun()

# Add a script to auto-refresh the page every 10 seconds, but only if not in history view
if not st.session_state.active_history_camera:
    st.markdown(
        """
        <script>
            setTimeout(function() {
                window.location.reload();
            }, 10000);  // 10 seconds in milliseconds
        </script>
        """,
        unsafe_allow_html=True
    )

# Initialize last detection time if not set
if 'last_detection' not in st.session_state:
    st.session_state.last_detection = get_current_time_ist() - timedelta(seconds=st.session_state.detection_interval)
    st.session_state.frame_counter = 0

# Check if enough time has passed for refresh
current_time = get_current_time_ist()
time_diff = (current_time - st.session_state.last_refresh).total_seconds()

# Update frames if refresh is due
if time_diff >= refresh_rate:
    # Update the last refresh time
    st.session_state.last_refresh = current_time

    # Process camera frames using the optimized module
    from camera_processing import process_camera_frames
    st.session_state.cameras = process_camera_frames(st.session_state.cameras, save_occupancy_data)

# Display cameras - two per row
if st.session_state.cameras:
    st.subheader("Camera Feeds")

    # Calculate number of rows needed
    num_cameras = len(st.session_state.cameras)
    num_rows = (num_cameras + 1) // 2  # Round up division

    for row in range(num_rows):
        cols = st.columns(2)

        # First camera in row
        idx = row * 2
        if idx < num_cameras:
            with cols[0]:
                camera = st.session_state.cameras[idx]

                # Camera title with view history button
                title_col1, title_col2 = st.columns([3, 1])
                with title_col1:
                    st.markdown(f"### {camera['name']}")
                with title_col2:
                    if st.button("View History", key=f"history_{idx}"):
                        st.session_state.active_history_camera = camera['camera_id']
                        st.session_state.selected_date = None  # Reset date selection
                        st.rerun()

                status = st.empty()
                frame_place = st.empty()

                # Show status
                if camera["status"] == "Connected":
                    status.success("Connected")
                else:
                    status.warning(camera["status"])

                # Display frame if available
                if camera["last_frame"] is None:
                    # Try to get first frame
                    frame_data, error = get_mjpeg_frame(camera['url'])
                    if frame_data:
                        st.session_state.cameras[idx]["last_frame"] = frame_data
                        st.session_state.cameras[idx]["status"] = "Connected"
                        image = Image.open(io.BytesIO(frame_data))
                        frame_place.image(image, use_column_width=True)
                        status.success("Connected")
                    else:
                        status.error(f"Failed to get frame: {error}")
                else:
                    # Display the cached frame
                    try:
                        image = Image.open(io.BytesIO(camera["last_frame"]))
                        frame_place.image(image, use_column_width=True)
                    except Exception as e:
                        status.error(f"Error displaying image: {str(e)}")

                # Create 3 columns for the detection buttons
                col1, col2, col3 = st.columns(3)

                # Occupancy detection toggle
                detection_active = camera.get('detection_active', False)
                with col1:
                    if st.button(
                        "Stop Occupancy" if detection_active else "Start Occupancy",
                        key=f"detect_btn_{idx}"
                    ):
                        # Toggle detection state
                        st.session_state.cameras[idx]['detection_active'] = not detection_active
                        # Save changes to JSON file
                        save_cameras(st.session_state.cameras)
                        st.rerun()

                # No-access detection toggle
                no_access_active = camera.get('no_access_active', False)
                with col2:
                    # Check if camera is in cooldown
                    camera_id = camera.get('camera_id')
                    in_cooldown = False
                    cooldown_text = ""

                    if 'no_access_cooldowns' in st.session_state and camera_id in st.session_state.no_access_cooldowns:
                        cooldown_info = st.session_state.no_access_cooldowns[camera_id]
                        current_time = get_current_time_ist()

                        if cooldown_info.get('in_cooldown', False) and current_time < cooldown_info.get('cooldown_until', current_time):
                            in_cooldown = True
                            remaining_seconds = (cooldown_info['cooldown_until'] - current_time).total_seconds()
                            remaining_minutes = int(remaining_seconds // 60)
                            remaining_seconds = int(remaining_seconds % 60)
                            cooldown_text = f" ({remaining_minutes}m)"

                    # Show button with cooldown status if applicable
                    button_text = "Stop No-Access" if no_access_active else "Start No-Access"
                    if in_cooldown and no_access_active:
                        button_text += cooldown_text

                    if st.button(button_text, key=f"no_access_btn_{idx}"):
                        # Toggle no-access detection state
                        st.session_state.cameras[idx]['no_access_active'] = not no_access_active
                        # Save changes to JSON file
                        save_cameras(st.session_state.cameras)
                        st.rerun()

                # Tailgating detection toggle
                tailgating_active = camera.get('tailgating_active', False)
                with col3:
                    # Check if camera is in cooldown
                    camera_id = camera.get('camera_id')
                    in_cooldown = False
                    cooldown_text = ""

                    if 'tailgating_cooldowns' in st.session_state and camera_id in st.session_state.tailgating_cooldowns:
                        cooldown_info = st.session_state.tailgating_cooldowns[camera_id]
                        current_time = get_current_time_ist()

                        if cooldown_info.get('in_cooldown', False) and current_time < cooldown_info.get('cooldown_until', current_time):
                            in_cooldown = True
                            remaining_seconds = (cooldown_info['cooldown_until'] - current_time).total_seconds()
                            remaining_minutes = int(remaining_seconds // 60)
                            remaining_seconds = int(remaining_seconds % 60)
                            cooldown_text = f" ({remaining_minutes}m)"

                    # Show button with cooldown status if applicable
                    button_text = "Stop Tailgating" if tailgating_active else "Start Tailgating"
                    if in_cooldown and tailgating_active:
                        button_text += cooldown_text

                    if st.button(button_text, key=f"tailgating_btn_{idx}"):
                        # Toggle tailgating detection state
                        st.session_state.cameras[idx]['tailgating_active'] = not tailgating_active
                        # Save changes to JSON file
                        save_cameras(st.session_state.cameras)
                        st.rerun()

        # Second camera in row
        idx = row * 2 + 1
        if idx < num_cameras:
            with cols[1]:
                camera = st.session_state.cameras[idx]

                # Camera title with view history button
                title_col1, title_col2 = st.columns([3, 1])
                with title_col1:
                    st.markdown(f"### {camera['name']}")
                with title_col2:
                    if st.button("View History", key=f"history_{idx}"):
                        st.session_state.active_history_camera = camera['camera_id']
                        st.session_state.selected_date = None  # Reset date selection
                        st.rerun()

                status = st.empty()
                frame_place = st.empty()

                # Show status
                if camera["status"] == "Connected":
                    status.success("Connected")
                else:
                    status.warning(camera["status"])

                # Display frame if available
                if camera["last_frame"] is None:
                    # Try to get first frame
                    frame_data, error = get_mjpeg_frame(camera['url'])
                    if frame_data:
                        st.session_state.cameras[idx]["last_frame"] = frame_data
                        st.session_state.cameras[idx]["status"] = "Connected"
                        image = Image.open(io.BytesIO(frame_data))
                        frame_place.image(image, use_column_width=True)
                        status.success("Connected")
                    else:
                        status.error(f"Failed to get frame: {error}")
                else:
                    # Display the cached frame
                    try:
                        image = Image.open(io.BytesIO(camera["last_frame"]))
                        frame_place.image(image, use_column_width=True)
                    except Exception as e:
                        status.error(f"Error displaying image: {str(e)}")

                # Create 3 columns for the detection buttons
                col1, col2, col3 = st.columns(3)

                # Occupancy detection toggle
                detection_active = camera.get('detection_active', False)
                with col1:
                    if st.button(
                        "Stop Occupancy" if detection_active else "Start Occupancy",
                        key=f"detect_btn_{idx}"
                    ):
                        # Toggle detection state
                        st.session_state.cameras[idx]['detection_active'] = not detection_active
                        # Save changes to JSON file
                        save_cameras(st.session_state.cameras)
                        st.rerun()

                # No-access detection toggle
                no_access_active = camera.get('no_access_active', False)
                with col2:
                    # Check if camera is in cooldown
                    camera_id = camera.get('camera_id')
                    in_cooldown = False
                    cooldown_text = ""

                    if 'no_access_cooldowns' in st.session_state and camera_id in st.session_state.no_access_cooldowns:
                        cooldown_info = st.session_state.no_access_cooldowns[camera_id]
                        current_time = get_current_time_ist()

                        if cooldown_info.get('in_cooldown', False) and current_time < cooldown_info.get('cooldown_until', current_time):
                            in_cooldown = True
                            remaining_seconds = (cooldown_info['cooldown_until'] - current_time).total_seconds()
                            remaining_minutes = int(remaining_seconds // 60)
                            remaining_seconds = int(remaining_seconds % 60)
                            cooldown_text = f" ({remaining_minutes}m)"

                    # Show button with cooldown status if applicable
                    button_text = "Stop No-Access" if no_access_active else "Start No-Access"
                    if in_cooldown and no_access_active:
                        button_text += cooldown_text

                    if st.button(button_text, key=f"no_access_btn_{idx}"):
                        # Toggle no-access detection state
                        st.session_state.cameras[idx]['no_access_active'] = not no_access_active
                        # Save changes to JSON file
                        save_cameras(st.session_state.cameras)
                        st.rerun()

                # Tailgating detection toggle
                tailgating_active = camera.get('tailgating_active', False)
                with col3:
                    # Check if camera is in cooldown
                    camera_id = camera.get('camera_id')
                    in_cooldown = False
                    cooldown_text = ""

                    if 'tailgating_cooldowns' in st.session_state and camera_id in st.session_state.tailgating_cooldowns:
                        cooldown_info = st.session_state.tailgating_cooldowns[camera_id]
                        current_time = get_current_time_ist()

                        if cooldown_info.get('in_cooldown', False) and current_time < cooldown_info.get('cooldown_until', current_time):
                            in_cooldown = True
                            remaining_seconds = (cooldown_info['cooldown_until'] - current_time).total_seconds()
                            remaining_minutes = int(remaining_seconds // 60)
                            remaining_seconds = int(remaining_seconds % 60)
                            cooldown_text = f" ({remaining_minutes}m)"

                    # Show button with cooldown status if applicable
                    button_text = "Stop Tailgating" if tailgating_active else "Start Tailgating"
                    if in_cooldown and tailgating_active:
                        button_text += cooldown_text

                    if st.button(button_text, key=f"tailgating_btn_{idx}"):
                        # Toggle tailgating detection state
                        st.session_state.cameras[idx]['tailgating_active'] = not tailgating_active
                        # Save changes to JSON file
                        save_cameras(st.session_state.cameras)
                        st.rerun()
else:
    st.info("No cameras added yet. Please add a camera using the form above.")

# Display history graphs if a camera is selected
if st.session_state.active_history_camera:
    # Find the camera name for the selected camera ID
    camera_name = "Unknown Camera"
    for cam in st.session_state.cameras:
        if cam.get('camera_id') == st.session_state.active_history_camera:
            camera_name = cam['name']
            break

    st.markdown("---")
    st.subheader(f"History for {camera_name}")

    # Add date selector
    available_dates = get_available_dates(st.session_state.active_history_camera)

    # Format dates for display
    date_options = ["Last 24 Hours"]
    if available_dates and len(available_dates) > 0:
        for d in available_dates:
            if isinstance(d, date):
                date_options.append(d.strftime("%Y-%m-%d"))
            else:
                # Just in case we get a string instead of a date object
                date_options.append(str(d))

    date_col1, date_col2, date_col3 = st.columns([2, 1, 1])

    with date_col1:
        selected_date_str = st.selectbox("Select Date", date_options, index=0)

        if selected_date_str == "Last 24 Hours":
            st.session_state.selected_date = None
        else:
            st.session_state.selected_date = datetime.strptime(selected_date_str, "%Y-%m-%d").date()

    with date_col3:
        if st.button("Close History View"):
            st.session_state.active_history_camera = None
            st.rerun()

    # Create tabs for different types of history
    tab1, tab2, tab3 = st.tabs(["Occupancy History", "No-Access Events", "Tailgating Events"])

    with tab1:
        # Create graphs - display one above the other to avoid congestion
        # First graph - Circular graph
        circular_fig = create_circular_graph(st.session_state.active_history_camera, st.session_state.selected_date)
        if circular_fig:
            st.plotly_chart(circular_fig, use_container_width=False)
        else:
            st.info("No occupancy data available for this camera on the selected date.")

        # Add some vertical space between graphs
        st.markdown("<br>", unsafe_allow_html=True)

        # Second graph - Hourly graph
        hourly_fig = create_hourly_graph(st.session_state.active_history_camera, st.session_state.selected_date)
        if hourly_fig:
            st.plotly_chart(hourly_fig, use_container_width=False)
        else:
            st.info("No occupancy data available for this camera on the selected date.")

    with tab2:
        # Display no-access events in tabular format
        from no_access_view import display_no_access_events
        display_no_access_events(st.session_state.active_history_camera, st.session_state.selected_date)

    with tab3:
        # Display tailgating events in tabular format
        from tailgating_view import display_tailgating_events
        display_tailgating_events(st.session_state.active_history_camera, st.session_state.selected_date)

# Add note about persistence
st.markdown("---")
st.caption("Camera settings are saved in 'cameras.json', occupancy data in 'occupancy_history.json', no-access events in 'no_access_events.json', and tailgating events in 'tailgating_events.json'. All times are in Indian Standard Time (IST, UTC+5:30).")

# Update every few seconds but avoid complete page rerun
time.sleep(1)  # Short sleep
