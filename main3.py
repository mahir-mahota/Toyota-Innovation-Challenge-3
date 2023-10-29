import datetime
from ultralytics import YOLO
import cv2


# initialize the video capture object
video_cap = cv2.VideoCapture("WIN_20231029_10_29_39_Pro.mp4")
# initialize the video writer object

# load the pre-trained YOLOv8n model
model = YOLO("last copy 2.pt")

while True:
    ret, frame = video_cap.read()
    if not ret:
      break
    results = model.track(frame, persist=True, conf = 0.01)
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    holes = (results[0].boxes.xyxy.cpu().numpy()).tolist()
    # show the frame to our screen
    cv2.imshow("Frame", annotated_frame)
    if cv2.waitKey(1) is ord('q'):
      break

video_cap.release()
cv2.destroyAllWindows()