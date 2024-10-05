import streamlit as st
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

st.title('YOLO Image and Video Processing')

# Allow users to upload images and videos
upload_file = st.file_uploader('Upload an image or video', type=['jpg','jpeg','png','bmp', 'mp4','mov','mkv'])

# Load YOLO model

try:
    model = YOLO('/home/sunny/Desktop/Car-Plate-Detection/license_plate_model.pt')
except Exception as e:
    st.error(f'Error In Loading YOLO model: {e}')


def predict_and_save_image(path_test_car, output_image_path):
    '''
    predict and save the bounding boxes on the given test image using the YOLO model

    Parameters:
    path_test_car(str): path to test image file
    output_image_path : path to save the test image file

    Return:
    return the path to saved image file
    '''

    try:
        results = model.predict(path_test_car)
        image = cv2.imread(path_test_car)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255,0) ,2)
                cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, image)
        return output_image_path
    except Exception as e:
        st.error(f'Error in Processing image: {e}')
        return None

def predict_and_show_video(video_path, output_path):
    '''

    Predict and save the bounding boxes on the given test video using the trained YOLO model 

    Parameters:
    video_path(str) : Path to the test video file.
    output_path(str): Path to save the output video file.

    Returns:
    str: The path to saved output video file

    '''
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f'Error opening video file: {video_path}')
            return None
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # FOURCC id 4-byte code used to specify the video codec. and video code on wikipedia
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            # ret means current frame are read successfully and frame is actual frame
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame, device='cpu')
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cv2.rectangle(frame, (x1, y1),(x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{confidence*100}%', (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            out.write(frame)
        cap.release()
        out.release()
        return output_path
    except Exception as e:
        st.error(f'Error in Processing image: {e}')
        return None

    

def process_media(input_path, output_path):
    '''
    Process the uploaded media file(image or video) and return the path to saved in the output file.

    Parameters:
    input_path (str) : path to the input media file.
    output_path(str) : path to save the output media file.

    Returns:
    str: The path to the saved output media file.
    '''

    file_extension = os.path.splitext(input_path)[1].lower()
    print(file_extension)
    if file_extension in ['.mp4','.avi', '.mov', '.mkv']:
        return predict_and_show_video(input_path, output_path)
    
    elif file_extension in ['.jpg','.jpeg','.png','.bmp']:
        return predict_and_save_image(input_path, output_path)
    else:
        st.error(f'Unsupported file type: {file_extension}')
        return None
    
if upload_file is not None:
    input_path = os.path.join('temp', upload_file.name)
    output_path= os.path.join('temp', f'output_{upload_file.name}')

    try:
        with open(input_path, 'wb') as f:
            f.write(upload_file.getbuffer())

        st.write('Processing......')
        result_path = process_media(input_path, output_path)
        if result_path:
            if input_path.endswith(('.mp4', '.avi','.mov', '.mkv')):
                print('yes')
            #     video_file = open(result_path, 'rb')
            #     video_bytes= video_file.read()
                st.video(result_path)
            else:
                st.image(result_path)
        print('ok')

    except Exception as e:
        st.error(f'Error In Uploading Or Processing File: {e}')
