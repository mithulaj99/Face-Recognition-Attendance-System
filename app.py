import streamlit as st
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import numpy as np
import shutil
import time

# -----------------------------
# Config
# -----------------------------
DATASET_PATH = "dataset"
MODEL_PATH = "face_recognition_model.pth"
ATTENDANCE_CSV = "attendance.csv"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load YOLO face detector (auto download)
@st.cache_resource
def load_face_detector():
    model_path = hf_hub_download(repo_id="Bingsu/adetailer", filename="face_yolov8n.pt")
    return YOLO(model_path)

face_detector = load_face_detector()

# -----------------------------
# Dataset Class
# -----------------------------
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = []

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx
                self.idx_to_class.append(class_name)
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith('.jpg'):
                        self.samples.append((os.path.join(class_dir, img_name), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# -----------------------------
# Train Function
# -----------------------------
def train_model(epochs=8, batch_size=8, lr=0.001):
    with st.spinner("Loading dataset..."):
        dataset = FaceDataset(root_dir=DATASET_PATH, transform=transform)
        if len(dataset) == 0:
            st.error("No images found! Add students first.")
            return

        num_classes = len(dataset.idx_to_class)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        progress_bar.progress((epoch + 1) / epochs)

    torch.save({
        'model_state_dict': model.state_dict(),
        'idx_to_class': dataset.idx_to_class
    }, MODEL_PATH)
    st.success("🎉 Model trained and saved!")

# -----------------------------
# Load Recognition Model
# -----------------------------
@st.cache_resource
def load_recognition_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found! Train first.")
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    num_classes = len(checkpoint['idx_to_class'])
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint['idx_to_class']

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Face Attendance System", layout="centered")
st.title("🧑‍🎓 Face Recognition Attendance System")

menu = ["Add Student", "Train Model", "Mark Attendance", "View Attendance"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Add Student":
    st.header("Add New Student")
    name = st.text_input("Enter Student Name")
    if st.button("Start Capturing (50 images)"):
        if not name:
            st.error("Enter a name!")
        else:
            student_dir = os.path.join(DATASET_PATH, name)
            os.makedirs(student_dir, exist_ok=True)
            if len(os.listdir(student_dir)) >= 50:
                st.warning("This student already has images. Adding more...")
            
            cap = cv2.VideoCapture(0)
            count = len(os.listdir(student_dir))
            stframe = st.empty()
            progress = st.progress(0)

            while count < len(os.listdir(student_dir)) + 50:
                ret, frame = cap.read()
                if not ret:
                    break

                results = face_detector(frame, conf=0.4)
                largest_box = None
                largest_area = 0
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            area = (x2 - x1) * (y2 - y1)
                            if area > largest_area:
                                largest_area = area
                                largest_box = (x1, y1, x2, y2)

                if largest_box:
                    x1, y1, x2, y2 = largest_box
                    margin = 0.2
                    w, h = x2 - x1, y2 - y1
                    x1 = max(0, int(x1 - margin * w))
                    y1 = max(0, int(y1 - margin * h))
                    x2 = min(frame.shape[1], int(x2 + margin * w))
                    y2 = min(frame.shape[0], int(y2 + margin * h))
                    face = frame[y1:y2, x1:x2]
                    face = cv2.resize(face, (224, 224))
                    cv2.imwrite(os.path.join(student_dir, f"{count}.jpg"), face)
                    count += 1
# Recommended: use fraction between 0.0 and 1.0
                    total_steps = len(os.listdir(student_dir)) + 50  # files + extra buffer
                    progress_value = min(count / total_steps, 1.0)   # never go over 1.0

                    progress.progress(progress_value)
                stframe.image(frame, channels="BGR")
                time.sleep(0.5)

            cap.release()
            st.success(f"✅ Captured 50 images for {name}!")

elif choice == "Train Model":
    st.header("Train Recognition Model")
    if st.button("Start Training (8 epochs)"):
        train_model()

elif choice == "Mark Attendance":
    st.header("Mark Attendance")
    model, idx_to_class = load_recognition_model()
    if model is None:
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(ATTENDANCE_CSV):
        pd.DataFrame(columns=['Name', 'Timestamp']).to_csv(ATTENDANCE_CSV, index=False)

    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = face_detector(frame, conf=0.4)
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size == 0:
                            continue

                        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                        face_tensor = transform(face_pil).unsqueeze(0).to(device)

                        with torch.no_grad():
                            outputs = model(face_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)
                            conf = confidence.item()

                            if conf > 0.8:
                                name = idx_to_class[predicted.item()]
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                df = pd.read_csv(ATTENDANCE_CSV)
                                today = timestamp[:10]
                                if not ((df['Name'] == name) & (df['Timestamp'].str.startswith(today))).any():
                                    new_row = pd.DataFrame({'Name': [name], 'Timestamp': [timestamp]})
                                    new_row.to_csv(ATTENDANCE_CSV, mode='a', header=False, index=False)
                                    st.success(f"✅ Marked: {name} at {timestamp}")

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                cv2.putText(frame, f"{name} ({conf:.2f})", (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            else:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(frame, f"Unknown ({conf:.2f})", (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            stframe.image(frame, channels="BGR")

        cap.release()

elif choice == "View Attendance":
    st.header("Attendance Records")
    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "attendance.csv", "text/csv")
    else:
        st.info("No attendance records yet.")