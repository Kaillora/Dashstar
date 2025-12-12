import cv2
from ultralytics import YOLO

# 1. Load the model
# We use 'yolov8n.pt' (Nano) because it is fast enough for a quick test
print("Loading model... (This might take a minute to download)")
model = YOLO('yolov8n.pt')

# 2. Define the GStreamer Pipeline for Jetson CSI Camera
# This tells OpenCV how to talk to the raw hardware sensor
gstreamer_str = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
)

# 3. Start Video Capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Could not open camera. Check your ribbon cable or running apps.")
    exit()

print("✅ Camera Opened! Press 'Q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Run Inference
    # verbose=False keeps the terminal clean
    results = model(frame, verbose=False)

    # 5. Show Results
    # This draws the boxes on the image
    annotated_frame = results[0].plot()

    # Calculate and print FPS just to see the baseline
    # (It will likely be slow, ~10-15 FPS. This is normal for Python!)
    cv2.imshow("YOLOv8 Sanity Check", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()