import torch
import cv2

# Load model
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

# Load class names from classes.txt
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Open the video file
cap = cv2.VideoCapture("test traffic video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Parse results
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
