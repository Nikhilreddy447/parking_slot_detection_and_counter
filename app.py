from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
from utils import get_parking_spots_bboxes, empty_or_not
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        video_file = request.files['video']
        mask_file = request.files['mask']
        
        if video_file and mask_file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4')
            mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_mask.png')
            
            video_file.save(video_path)
            mask_file.save(mask_path)
            
            processed_video_path, empty_spots = process_video(video_path, mask_path)
            return render_template('result.html', empty_spots=empty_spots, video_path=processed_video_path)
        
    return redirect(url_for('index'))

def process_video(video_path, mask_path):
    mask = cv2.imread(mask_path, 0)
    video = cv2.VideoCapture(video_path)

    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    parking_spots = get_parking_spots_bboxes(connected_components)

    if not video.isOpened():
        print("Error: Could not open video.")
        return

    spots_status = [None for j in parking_spots]
    diffs = [None for j in parking_spots]
    step = 30
    frame_nmr = 0
    previous_frame = None

    empty_spots_list = []

    # Video writer
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    processed_video_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_video.mp4')
    out = cv2.VideoWriter(processed_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        if frame_nmr % step == 0 and previous_frame is not None:
            for spot_idx, spot in enumerate(parking_spots):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                diffs[spot_idx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])
        
        if frame_nmr % step == 0:
            if previous_frame is None:
                arr_ = range(len(parking_spots))
            else:
                arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]

            for spot_idx in arr_:
                spot = parking_spots[spot_idx]
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                spot_status = empty_or_not(spot_crop)
                spots_status[spot_idx] = spot_status

            empty_spots_list.append(sum(spots_status))

        if frame_nmr % step == 0:
            previous_frame = frame.copy()

        for spot_idx, spot in enumerate(parking_spots):
            spot_status = spots_status[spot_idx]
            x1, y1, w, h = parking_spots[spot_idx]
            if spot_status:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 225, 0), 2)
            else:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 225), 2)

        cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
        cv2.putText(frame, 'Available spots : {}/{}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

        frame_nmr += 1

    video.release()
    out.release()
    
    return processed_video_path, empty_spots_list

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

@app.route('/processed/<filename>')
def send_video(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
