# Face and Eye Blink Detection

This repository implements a real-time **Face Detection and Eye Blink Detection System** using **Mediapipe** and **OpenCV**. The system is designed to detect faces and monitor eye blinks efficiently, making it suitable for applications such as drowsiness detection, driver monitoring, and human-computer interaction.

## Features

### 1. Real-Time Face Detection
- Utilizes Mediapipe's **Face Mesh** module to detect and track facial landmarks with high precision.
- Optimized for real-time performance, even on low-power devices.

### 2. Eye Blink Detection
- Detects eye blinks using key eye landmarks provided by Mediapipe.
- Calculates the Eye Aspect Ratio (EAR) to determine the eye's state (open or closed).
- Tracks and counts blinks based on EAR variations over time.

### 3. OpenCV Integration
- OpenCV is used for:
  - Capturing video from webcam feeds.
  - Preprocessing video frames.
  - Overlaying visual indicators such as bounding boxes and status messages.

### 4. Customizability
- Adjustable EAR thresholds and blink duration parameters to suit different user needs.
- Can be extended for detecting additional facial expressions or gestures.

## Applications
- **Driver Drowsiness Detection**: Alerts drivers if signs of fatigue or drowsiness are detected.
- **Health Monitoring**: Tracks user activity and fatigue levels in healthcare scenarios.
- **Human-Computer Interaction**: Enables hands-free interaction using eye blinks as input signals.

## Technical Stack
- **Mediapipe**: For facial landmark detection and tracking.
- **OpenCV**: For video processing and visualizations.
- **Python**: The programming language used for the implementation.

## Installation

1. Clone the repository:
   ```bash
   https://github.com/AlphaDweb/eye-detection.git
   cd Face-Eye-Blink-Detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project
   
   i. For face detection:
   ```bash
   python FaceDetection.py
   ```
   ii. For eye blink detection:
   ```bash
   python EyeBlinkDetection.py
   ```

## Usage
- Ensure you have a webcam connected to your device.
- Run the `FaceDetection.py` or `EyeBlinkDetection.py` script to start the detection system.
- The application will display a live video feed with detected faces and eye blink status.

## Contributing
Contributions are welcome! If you would like to enhance the project or fix any issues, feel free to submit a pull request.

## Acknowledgments
- [Mediapipe](https://mediapipe.dev) for providing a robust framework for facial landmark detection.
- [OpenCV](https://opencv.org) for video processing and visualization tools.
