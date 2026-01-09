from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

app = FastAPI()
device = torch.device("cpu")

model = DigitCNN()
model.load_state_dict(torch.load("DigitModel.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

@app.get("/", response_class=HTMLResponse)
def home():
    return open("index.html").read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes))

    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, 1)

    return JSONResponse({"predict": int(pred.item())})
