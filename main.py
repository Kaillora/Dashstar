import cv2
from ultralytics import YOLO

# 1. Load the YOLOv8 Nano model
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')

# 2. Define the GStreamer Pipeline for the IMX219
# This is critical for the Jetson Orin Nano to see the camera via OpenCV
gstreamer_str = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=(int)1024, height=(int)600, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=(int)1024, height=(int)600, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
)

# 3. Open the Camera
cap = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

print("✅ Camera active. Press 'Q' to exit.")

while True:
    success, frame = cap.read()
    if success:
        # Run inference
        results = model(frame, verbose=False)

        # Draw the boxes
        annotated_frame = results[0].plot()

        # Show the frame
        cv2.imshow("Jetson YOLOv8", annotated_frame)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()