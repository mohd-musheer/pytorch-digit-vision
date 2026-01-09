🧠 PyTorch Digit Vision – Handwritten Digit Recognizer

A deep learning–powered web application that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with PyTorch, served via FastAPI, containerized using Docker, and deployed live on Render.

🚀 Live Demo

🔗 Web App:
👉 https://pytorch-digit-vision.onrender.com

🔗 Docker Image:
👉 https://hub.docker.com/repository/docker/mohdmusheer/pytorch-digit-vision

📸 What This Project Does

Accepts an image of a handwritten digit

Preprocesses it (grayscale, resize, normalize)

Feeds it into a CNN (Conv2D-based)

Predicts the digit (0–9) with high accuracy

Exposes the model via a REST API

🛠️ Tech Stack Used
🔹 Machine Learning

PyTorch – CNN model implementation

torchvision – image utilities

Pillow (PIL) – image preprocessing

NumPy – numerical operations

🔹 Backend / API

FastAPI – high-performance REST API

Uvicorn – ASGI server

🔹 DevOps / Deployment

Docker – containerization

Docker Hub – image hosting

Render – cloud deployment (CPU-based)

🧠 Model Architecture (CNN)
Input Image (1 × 28 × 28)
   ↓
Conv2D (1 → 32) + ReLU
   ↓
Conv2D (32 → 64) + ReLU
   ↓
MaxPooling (2×2)
   ↓
Flatten
   ↓
Fully Connected (128)
   ↓
Output Layer (10 classes)


The model is trained on handwritten digit images (MNIST-style) and saved as a .pth file.

📂 Project Structure
pytorch-digit-vision/
│
├── digitapi.py        # FastAPI app + CNN model
├── DigitModel.pth     # Trained PyTorch model
├── Dockerfile         # Docker configuration
├── requirements.txt  # Python dependencies
├── index.html         # Frontend UI
└── README.md

▶️ How to Use (Live App)

Open the live URL
👉 https://pytorch-digit-vision.onrender.com

Upload or draw a digit image

Submit the image

Get the predicted digit instantly

🐳 Run with Docker (Recommended)
1️⃣ Pull the Docker image
docker pull mohdmusheer/pytorch-digit-vision

2️⃣ Run the container
docker run -p 8000:8000 mohdmusheer/pytorch-digit-vision

3️⃣ Open in browser
http://localhost:8000

🧪 API Usage
Endpoint
POST /predict

Request

Content-Type: multipart/form-data

Body: Image file (handwritten digit)

Response
{
  "prediction": 7
}

⚙️ Run Locally (Without Docker)
1️⃣ Clone the repo
git clone https://github.com/mohd-musheer/pytorch-digit-vision.git
cd pytorch-digit-vision

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Start the server
uvicorn digitapi:app --reload

4️⃣ Open
http://127.0.0.1:8000

📈 Key Learnings from This Project

How Conv2D works in real CNNs

Image preprocessing for neural networks

Serving ML models using FastAPI

Dockerizing PyTorch applications

Deploying ML apps on cloud platforms

Handling model loading safely in production

🎯 Future Improvements

Add confidence score to predictions

Support drawing canvas input

GPU deployment option

Batch prediction support

Model versioning

👤 Author

Musheer
Machine Learning & Deep Learning Enthusiast
Focused on PyTorch, Computer Vision & Deployment