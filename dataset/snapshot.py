import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

CLASSES = ["compost", "recycle", "trash"]

# load model
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, 3)
model.load_state_dict(torch.load("waste_model_1.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(1)

print("Press SPACE to take photo, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)

    # SPACE = snapshot
    if key == 32:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            out = model(img)
            probs = torch.softmax(out, dim=1)[0]
            pred = probs.argmax().item()

        print("\nPrediction:", CLASSES[pred])
        for i, cls in enumerate(CLASSES):
            print(f"  {cls}: {probs[i]*100:.2f}%")

    # Q = quit
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
