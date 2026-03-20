import os
import time
import cv2
import torch
import serial
import gpiod
from gpiod.line import Direction
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = os.path.expanduser("~/waste-env/waste_model_1.pth")
CLASSES = ["compost", "recycle", "trash"]
IR_SENSOR_PIN = 17
CAPTURE_DELAY = 0.5       # wait for object to settle before taking pic
COOLDOWN = 2.0             # wait between detections so it doesn't spam

# connect to arduino over usb serial
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)  # give arduino time to reset

# image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# load the trained model
print("Loading model...")
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
print("Model loaded.")

# set up the IR sensor on GPIO 17
request = gpiod.request_lines(
    "/dev/gpiochip4",
    consumer="waste-sorter",
    config={IR_SENSOR_PIN: gpiod.LineSettings(direction=Direction.INPUT)}
)

def classify_frame(frame):
    """take a camera frame, run it through the model, return the label and confidence"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred = probs.argmax().item()
    return CLASSES[pred], probs[pred].item()

# open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit(1)

print("System ready. Waiting for waste...")

try:
    while True:
        # check if something is in front of the IR sensor
        if request.get_value(IR_SENSOR_PIN) == gpiod.line.Value.ACTIVE:
            print("Object detected!")
            time.sleep(CAPTURE_DELAY)

            ret, frame = cap.read()
            if not ret:
                print("Camera capture failed")
                continue

            label, confidence = classify_frame(frame)
            print(f"  -> {label} ({confidence*100:.1f}%)")

            # tell the arduino which bin to rotate to
            if label == "compost":
                arduino.write(b'1')
            elif label == "recycle":
                arduino.write(b'2')
            elif label == "trash":
                arduino.write(b'3')

            print(f"  -> Sent '{label}' command to Arduino")
            time.sleep(COOLDOWN)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Shutting down...")

finally:
    cap.release()
    arduino.close()
    request.release()
    print("Cleanup complete.")