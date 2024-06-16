import cv2
import numpy as np
from utils import get_parking_spots_bboxes,empty_or_not
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

mask_path = r'input_data\mask_1920_1080.png'
video_path = r'input_data\parking_1920_1080_loop.mp4'

def calc_diff(im1,im2):
    return np.abs(np.mean(im1) - np.mean(im2))

mask = cv2.imread(mask_path,0)
video = cv2.VideoCapture(video_path)

# Extracting the bounding boxes for the video with connected Components
connected_components = cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)

parking_spots = get_parking_spots_bboxes(connected_components)
'''
Used to display one of bounding boxes
print(parking_spots[0])
'''
# Defining the window for the ouput window
if not video.isOpened():
    print("Error: Could not open video.")
    exit()
window_name = "Parking video"
ret = True
bounding_box_edge_width = 2
spots_status = [None for j in parking_spots]
diffs = [None for j in parking_spots]

# We are using the step to clasify the slop for every sec, initaily the model trained on each frame that is 30 frames per sec. so 30 classifications per second. to restrict it to 1 classification per second
step = 30
frame_nmr = 0

previous_frame = None
while True:
    
    ret, frame = video.read()
    if not ret:
        print("Reached the end of the video.")
        break
    
    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_idx,spot in enumerate(parking_spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1+h,x1:x1+w,:]
            
            diffs[spot_idx] = calc_diff(spot_crop,previous_frame[y1:y1+h,x1:x1+w,:])
        '''plt.figure()
        plt.hist([diffs[j]/np.amax(diffs) for j in np.argsort(diffs)][::-1])
        if frame_nmr == 300:
            plt.show()'''
            
            
    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(parking_spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j]/np.amax(diffs) > 0.4 ]
            
        for spot_idx in arr_:
            spot = parking_spots[spot_idx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1+h,x1:x1+w,:]
            #model to predict is empty
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_idx] = spot_status
    
    if frame_nmr % step == 0:
        previous_frame = frame.copy()
        
    for spot_idx,spot in enumerate(parking_spots):
        spot_status = spots_status[spot_idx]
        x1, y1, w, h = parking_spots[spot_idx]
        if spot_status:
            frame = cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,225,0),bounding_box_edge_width)
        else:
            frame = cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,0,225),bounding_box_edge_width)

    cv2.rectangle(frame,(80,20),(550,80),(0,0,0),-1)
    cv2.putText(frame,'Available spots : {}/{}'.format(str(sum(spots_status)),str(len(spots_status))),(100,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)
    cv2.resizeWindow(window_name, 800, 600)
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_nmr += 1
    


