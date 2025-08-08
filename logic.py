
import torch
import cv2

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Load video
cap = cv2.VideoCapture('test traffic video.mp4')  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results.render()[0]

    # Show frame
    cv2.imshow("YOLOv5 Vehicle Detection", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
