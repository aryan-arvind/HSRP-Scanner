import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sort.sort import *
from util import get_car, read_license_plate, write_csv

# Function to handle video upload and processing
def process_video(video_file):
    # Initialize tracker and models
    mot_tracker = Sort()
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('./models/license_plate_detector.pt')
    
    # Create a dictionary to store results
    results = {}
    
    # OpenCV video capture
    cap = cv2.VideoCapture(video_file)

    # Define vehicles classes to detect
    vehicles = [2, 3, 5, 7]
    
    # Read frames and process them
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            # Detect vehicles using YOLO
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            # Track vehicles using SORT
            track_ids = mot_tracker.update(np.asarray(detections_))

            # Detect license plates using YOLO
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # Assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # Crop license plate and process
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # Read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
    
    # Write results to CSV
    write_csv(results, './output.csv')

    return results

# Streamlit UI components
st.title("Vehicle and License Plate Detection")
st.markdown("Upload a video file for vehicle and license plate detection.")

# File upload widget
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file, format="video/mp4", start_time=0)

    # Process the uploaded video
    st.button("Start Detection")

    if st.button("Start Detection"):
        # Convert the uploaded file to a format suitable for OpenCV
        video_file = uploaded_file.name
        with open(video_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the video file
        results = process_video(video_file)
        
        # Display results in a DataFrame
        st.header("Detection Results")
        st.write("Below is the data of detected vehicles and license plates:")
        
        # Convert results to a DataFrame (for easy display)
        result_data = []
        for frame, frame_data in results.items():
            for car_id, car_data in frame_data.items():
                result_data.append({
                    "Frame": frame,
                    "Car ID": car_id,
                    "Vehicle BBox": car_data["car"]["bbox"],
                    "License Plate BBox": car_data["license_plate"]["bbox"],
                    "License Plate Text": car_data["license_plate"]["text"],
                    "BBox Score": car_data["license_plate"]["bbox_score"],
                    "Text Score": car_data["license_plate"]["text_score"]
                })
        
        if result_data:
            df = pd.DataFrame(result_data)
            st.dataframe(df)
            
            # Provide download link for CSV
            csv = df.to_csv(index=False)
            st.download_button(label="Download Results as CSV", data=csv, file_name="detection_results.csv", mime="text/csv")
        else:
            st.write("No vehicles or license plates detected.")

else:
    st.warning("Please upload a video to start.")
