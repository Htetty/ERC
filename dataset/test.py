import torch
from torchvision import transforms, models
from PIL import Image

IMAGES = ["IMG_6457.jpeg", "IMG_6458.jpeg"]
CLASSES = ["compost", "recycle", "trash"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, 3)
model.load_state_dict(torch.load("waste_model.pth", map_location="cpu"))
model.eval()

for img_path in IMAGES:
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1)[0]
        pred = probs.argmax().item()

    print(f"\nImage: {img_path}")
    print(f"Prediction: {CLASSES[pred]}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {probs[i]*100:.2f}%")
