# NexusX – ISL Gesture Recognition (3rd Sem Project)

This project detects Indian Sign Language (A–Z) using image-based training and real-time gesture recognition through a webcam.  
It was built during Smart India Hackathon by Team **NexusX** (Shriraksha, Yukthi, Ayush, Amrutha) in 3rd semester.  
The model uses MediaPipe keypoints, TensorFlow, and OpenCV to identify hand gestures and output the predicted alphabet live on screen.

### ✨ Features
- Training from A–Z sign images  
- Keypoint extraction using MediaPipe  
- Model training with TensorFlow  
- Real-time prediction using webcam  
- Clean, simple college-level ML workflow  

### 📦 Requirements
tensorflow
opencv-python
mediapipe
scikit-learn
numpy


### 🚀 Running the Project
1. Create virtual environment  
2. Install dependencies from `requirements.txt`  
3. Generate keypoints (if needed)  
4. Train model using `trainmodel.py`  
5. Run real-time detection using `app.py`

### 📁 Dataset
Dataset (images + MP_Data keypoints) is **NOT stored in GitHub** because of large size.  
You can add your own dataset inside:

├── images/
├──Logs\train/
├── MP_Data/


### 👥 Team NexusX
- Shriraksha  
- Yukthi NT  
- Ayush  
- Amrutha  

This was our first machine-learning project built during SIH selection round.
