import cv2

video_path = 'input_data/parking_1920_1080_loop.mp4'

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

window_name = 'Praking Video Sample'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600) 
ret = True
while True:
    ret, frame = video.read()
    if not ret:
        print("Reached the end of the video.")
        break

    cv2.imshow(window_name, frame)
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
