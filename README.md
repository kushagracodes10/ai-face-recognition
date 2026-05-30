# AI Face Recognition System

A real-time AI-powered face recognition application developed using Python, OpenCV, MediaPipe, and NumPy.

## Overview

This project detects and recognizes faces from a live webcam feed. Known faces are stored in a dataset folder, and the system compares detected faces against the stored face embeddings to identify individuals in real time.

## Features

* Real-time webcam face detection
* Face recognition using stored face embeddings
* Multiple known face support
* Automatic face labeling
* Live video processing
* Fast and lightweight implementation

## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy

## Project Structure

```text
AI_Face_Recognition/
│
├── main.py
├── convert_image.py
├── requirements.txt
├── known_faces/
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/ai-face-recognition.git
```

2. Navigate to the project directory:

```bash
cd ai-face-recognition
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
python main.py
```

## How It Works

1. The system loads images from the `known_faces` directory.
2. Face embeddings are generated for each known individual.
3. Webcam video is processed frame by frame.
4. Faces are detected using MediaPipe.
5. Detected faces are compared against stored embeddings.
6. Matching faces are labeled with their corresponding names.

## Applications

* Attendance systems
* Smart surveillance
* Access control systems
* Identity verification
* Educational AI projects

## Future Improvements

* Deep learning-based face embeddings
* Database integration
* Higher recognition accuracy
* Web-based interface
* Multi-camera support

## Author

Kushagra Varshney

## License

This project is intended for educational and learning purposes.
