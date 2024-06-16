import cv2
from utils import get_parking_spots_bboxes
import warnings
warnings.filterwarnings('ignore')

mask_path = r'input_data\mask_crop.png'
video_path = r'input_data\parking_crop_loop.mp4'

mask = cv2.imread(mask_path,0)
video = cv2.VideoCapture(video_path)


# Extracting the bounding boxes for the video with connected Components

connected_components = cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)

parking_spots = get_parking_spots_bboxes(connected_components)


'''
Used to display one of bounding boxes
print(parking_spots[0])
'''
window_name = "Parking video"
ret = True
bounding_box_edge_width = 2
while True:
    ret, frame = video.read()
    if not ret:
        print("Reached the end of the video.")
        break
    
    for spot in parking_spots:
        x1, y1, w, h = spot
        frame = cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(225,0,0),bounding_box_edge_width)

    cv2.imshow(window_name, frame)
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
