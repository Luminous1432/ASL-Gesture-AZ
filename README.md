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

## Dataset

The dataset used in this project is available [here on Kaggle](https://www.kaggle.com/datasets/shreerakshagourayya/indian-sign-language-az-dataset-nexusx/data/data/data).

### Collection Methodology
- Videos of different ISL gestures were recorded using a webcam by all team members.  
- Each gesture was repeated multiple times to ensure variety and accuracy.  
- Keypoints for hands, face, and body were extracted using MediaPipe.  
- Data was labeled carefully to maintain consistency.  
- All preprocessing, extraction, and labeling were done by the VedaX team.

### Authors
- Shriraksha Gourayya – Data collection, keypoint extraction, project design  
- Ayush – Team leader, project coordination, code review  
- Yukthi – Data labeling and preprocessing  
- Amrutha – Video recording and dataset management  

**Team:** VedaX, Srinivas Institute of Technology

### Provenance
This dataset was fully collected and created by the VedaX team. All videos, labels, and keypoints are original, and no external sources were used.

### License
This dataset is licensed under **CC BY-NC-SA 4.0**. Non-commercial use only. Please give credit to the authors if you use it.


This was our first machine-learning project built during SIH selection round.
