# Face-Recognition-Attendance-System
A real-time, contactless attendance management system using Deep Learning and Computer Vision. The application detects faces using YOLOv8, recognizes individuals using a fine-tuned ResNet18 model, and records attendance automatically through a Streamlit web interface.

# Features
- Real-time face detection using YOLOv8

-  Face recognition using ResNet18 (Transfer Learning)

- Automated dataset creation via webcam (50 images per student)
 
- Duplicate-free daily attendance marking

- Confidence-based recognition (threshold > 80%)

- Attendance stored and exportable as CSV

- Interactive and user-friendly Streamlit UI

- CPU & GPU compatible (CUDA supported)


# Tech Stack

- Programming Language: Python

- Frontend: Streamlit

- Computer Vision: OpenCV, YOLOv8

- Deep Learning: PyTorch, Torchvision

- Model Architecture: ResNet18

- Data Handling: Pandas, NumPy

- Deployment Support: Hugging Face Hub (YOLO model download)


# Project Structure
├── app.py                      # Main Streamlit application
├── dataset/                    # Collected face images (per student)
│   ├── Student_1/
│   ├── Student_2/
├── face_recognition_model.pth  # Trained model checkpoint
├── attendance.csv              # Attendance records
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation

# Installation & Setup
1. Clone the Repository
git clone https://github.com/your-username/face-attendance-system.git
cd face-attendance-system

2. Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Application
streamlit run app.py


# How It Works
1. Add Student
- Enter student name

- Capture 50 face images automatically using webcam

- Images are stored in the dataset folder

2. Train Model
- Trains a ResNet18 classifier using transfer learning

- Saves model and class mappings locally

3. Mark Attendance
- Detects faces in real-time

- Recognizes known students

- Marks attendance once per day per student

4. View Attendance
- Displays attendance records
- Download CSV file for reports reports



# Model Details

- Face Detection: YOLOv8 (pretrained face model)

- Recognition Model: ResNet18

- Loss Function: CrossEntropyLoss

- Optimizer: Adam

- Input Size: 224 × 224

- Confidence Threshold: 0.80


# Use Cases

- Educational institutions
- Corporate training sessions
- Smart classrooms
- Secure access systems
- Research and academic projects

# Future Enhancements
- Database integration (MySQL / PostgreSQL)
- Role-based authentication
- Cloud deployment
- Liveness detection (anti-spoofing)
- Mobile/web camera support
- Attendance analytics dashboard

# Author

Mithulaj Vk
AI / Machine Learning Enthusiast


# License

This project is licensed under the MIT License.
Feel free to use, modify, and distribute with attribution.
