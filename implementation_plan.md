# Sleep Position Analysis System - Detailed Implementation Plan

## Technologies and Libraries Overview

### Core Libraries

| Library | Version | Purpose | Documentation |
|---------|---------|---------|---------------|
| Python | 3.8+ | Core programming language | [Python Docs](https://docs.python.org/3/) |
| OpenCV | 4.8.0.76 | Computer vision and image processing | [OpenCV Docs](https://docs.opencv.org/4.x/) |
| MediaPipe | 0.10.5 | Face and body detection/tracking | [MediaPipe Docs](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python) |
| NumPy | 1.24.3 | Numerical operations and data handling | [NumPy Docs](https://numpy.org/doc/) |
| SQLite | Built-in | Local database for position data storage | [SQLite Docs](https://www.sqlite.org/docs.html) |
| Matplotlib | Latest | Data visualization for reports | [Matplotlib Docs](https://matplotlib.org/stable/index.html) |

## Phase 1: Face Detection Demo

### Step 1: Development Environment Setup

#### 1.1 Python Installation

If Python is not already installed:

```bash
# Check if Python is installed and which version
python3 --version

# If not installed, on macOS use Homebrew
brew install python3
```

#### 1.2 Virtual Environment Setup

```bash
# Navigate to project directory
cd ~/projects/stay_over_there

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### 1.3 Install Required Libraries

Create a `requirements.txt` file with the following content:

```
opencv-python==4.8.0.76
mediapipe==0.10.5
numpy==1.24.3
matplotlib==3.7.1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Step 2: Camera Interface Implementation

#### 2.1 Basic Camera Access Module

Create a file named `camera_module.py` with the following content:

```python
import cv2
import time

class CameraInterface:
    def __init__(self, camera_id=0, width=640, height=480):
        """
        Initialize the camera interface.
        
        Args:
            camera_id: Camera device ID (default: 0 for built-in webcam)
            width: Desired frame width
            height: Desired frame height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.is_running = False
        self.last_frame = None
        self.fps = 0
        self.prev_frame_time = 0
    
    def start(self):
        """
        Start the camera capture.
        
        Returns:
            bool: True if camera started successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
            
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.is_running = True
        print(f"Camera started with resolution: {self.width}x{self.height}")
        return True
    
    def stop(self):
        """
        Stop the camera capture and release resources.
        """
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
            print("Camera stopped")
    
    def get_frame(self):
        """
        Capture a single frame from the camera.
        
        Returns:
            numpy.ndarray: The captured frame, or None if capture failed
            float: Current FPS
        """
        if not self.is_running:
            return None, 0
            
        ret, frame = self.cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            return None, 0
            
        # Calculate FPS
        current_time = time.time()
        if self.prev_frame_time > 0:
            self.fps = 1 / (current_time - self.prev_frame_time)
        self.prev_frame_time = current_time
        
        self.last_frame = frame
        return frame, self.fps
    
    def get_camera_properties(self):
        """
        Get camera properties.
        
        Returns:
            dict: Camera properties including resolution and FPS
        """
        if not self.is_running:
            return {}
            
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        return {
            "width": actual_width,
            "height": actual_height,
            "fps": actual_fps
        }
```

#### 2.2 Testing Camera Module

Create a file named `test_camera.py` to test the camera interface:

```python
import cv2
from camera_module import CameraInterface

def main():
    # Initialize camera
    camera = CameraInterface()
    if not camera.start():
        print("Failed to start camera")
        return
        
    # Print camera properties
    properties = camera.get_camera_properties()
    print(f"Camera properties: {properties}")
    
    try:
        # Create window
        cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
        
        while True:
            # Get frame
            frame, fps = camera.get_frame()
            
            if frame is None:
                break
                
            # Add FPS text
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("Camera Test", frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        camera.stop()
        cv2.destroyAllWindows()
        print("Test completed")

if __name__ == "__main__":
    main()
```

### Step 3: Face Detection Implementation

#### 3.1 MediaPipe Face Detection Module

Create a file named `face_detector.py` with the following content:

```python
import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        """
        Initialize the MediaPipe face detector.
        
        Args:
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
                detection to be considered successful. (default: 0.5)
        """
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range detection (within 2 meters)
            min_detection_confidence=min_detection_confidence
        )
        
    def detect_faces(self, frame):
        """
        Detect faces in the given frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            processed_frame: Frame with detection visualizations
            face_data: List of dictionaries containing face detection data
                Each dict contains: 'bbox' (x, y, w, h), 'score', 'landmarks'
        """
        if frame is None:
            return None, []
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Face Detection
        results = self.face_detection.process(rgb_frame)
        
        # Create a copy of the frame for drawing
        processed_frame = frame.copy()
        
        # List to store face data
        face_data = []
        
        # Check if any faces were detected
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox_rel = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bbox_rel.xmin * iw), int(bbox_rel.ymin * ih), \
                            int(bbox_rel.width * iw), int(bbox_rel.height * ih)
                
                # Draw bounding box
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Get confidence score
                score = detection.score[0]
                
                # Display confidence score
                confidence_text = f"{int(score * 100)}%"
                cv2.putText(processed_frame, confidence_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Extract face landmarks
                landmarks = []
                for i in range(6):  # MediaPipe provides 6 landmarks for each face
                    try:
                        landmark = detection.location_data.relative_keypoints[i]
                        lx, ly = int(landmark.x * iw), int(landmark.y * ih)
                        landmarks.append((lx, ly))
                        
                        # Draw landmark points
                        cv2.circle(processed_frame, (lx, ly), 5, (255, 0, 0), -1)
                    except IndexError:
                        continue
                
                # Store face data
                face_data.append({
                    'bbox': (x, y, w, h),
                    'score': score,
                    'landmarks': landmarks
                })
        
        # Display face count
        face_count = len(face_data)
        cv2.putText(processed_frame, f"Faces: {face_count}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return processed_frame, face_data
```

#### 3.2 Main Application

Create a file named `face_detection_app.py` that combines the camera module and face detector:

```python
import cv2
import time
from camera_module import CameraInterface
from face_detector import FaceDetector

def main():
    # Initialize camera
    camera = CameraInterface(width=1280, height=720)
    if not camera.start():
        print("Failed to start camera")
        return
    
    # Initialize face detector
    detector = FaceDetector(min_detection_confidence=0.5)
    
    # Print camera properties
    properties = camera.get_camera_properties()
    print(f"Camera properties: {properties}")
    
    try:
        # Create window
        cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
        
        while True:
            # Get frame
            frame, fps = camera.get_frame()
            
            if frame is None:
                break
            
            # Detect faces
            processed_frame, face_data = detector.detect_faces(frame)
            
            # Add FPS text
            cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add instructions
            cv2.putText(processed_frame, "Press 'q' to quit", 
                        (10, processed_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display the frame
            cv2.imshow("Face Detection", processed_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        camera.stop()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    main()
```

### Step 4: Testing and Refinement

#### 4.1 Testing Protocol

Create a file named `test_protocol.md` with testing instructions:

```markdown
# Face Detection Testing Protocol

## Basic Functionality Tests

1. **Camera Initialization**
   - Run `test_camera.py`
   - Verify camera starts without errors
   - Confirm video feed displays properly
   - Check FPS counter is working

2. **Face Detection Accuracy**
   - Run `face_detection_app.py`
   - Test with single face at different distances (0.5m, 1m, 2m)
   - Test with multiple faces in frame
   - Verify bounding boxes are correctly placed around faces
   - Check confidence scores are displayed

3. **Performance Testing**
   - Monitor FPS during operation
   - Target: Maintain at least 15 FPS
   - Note any performance degradation patterns

## Environmental Tests

1. **Lighting Conditions**
   - Test in bright lighting
   - Test in moderate lighting
   - Test in low lighting
   - Note detection accuracy in each condition

2. **Angles and Occlusion**
   - Test with face at different angles (front, 45°, profile)
   - Test with partial face occlusion (hand, hair, glasses)
   - Note detection limitations

## Refinement Checklist

- [ ] Optimize for consistent FPS
- [ ] Adjust detection confidence threshold if needed
- [ ] Improve visualization (colors, text size, etc.)
- [ ] Add additional information display if useful
```

## Data Models

### Face Detection Data Model

```python
# Face detection result structure
face_data = {
    'bbox': (x, y, w, h),  # Bounding box coordinates (top-left x, y, width, height)
    'score': 0.95,         # Confidence score (0.0 to 1.0)
    'landmarks': [         # List of facial landmarks as (x, y) coordinates
        (100, 120),        # Right eye
        (140, 120),        # Left eye
        (120, 140),        # Nose tip
        (100, 160),        # Right mouth corner
        (140, 160),        # Left mouth corner
        (120, 180)         # Mouth center
    ]
}
```

## Future Phases (Detailed)

### Phase 2: Bed Area Mapping

#### Technologies and Libraries
- OpenCV for image processing and user interface
- NumPy for coordinate calculations
- JSON for storing bed boundary configuration

#### Implementation Steps
1. Create a calibration interface to define bed boundaries
2. Implement perspective transformation for overhead view
3. Define and store bed sides division
4. Create visualization of bed boundaries

#### Sample Code for Bed Mapping Interface

```python
# Conceptual code for bed area mapping
class BedMapper:
    def __init__(self):
        self.bed_corners = []  # [top_left, top_right, bottom_right, bottom_left]
        self.dividing_line = []  # [(x1, y1), (x2, y2)]
        self.config_file = "bed_config.json"
    
    def calibrate(self, frame):
        # Interactive calibration process
        # User clicks to define bed corners and dividing line
        pass
    
    def save_configuration(self):
        # Save bed mapping to JSON file
        pass
    
    def load_configuration(self):
        # Load bed mapping from JSON file
        pass
    
    def transform_to_overhead_view(self, frame):
        # Apply perspective transform for top-down view
        pass
    
    def visualize_bed_boundaries(self, frame):
        # Draw bed boundaries and dividing line on frame
        pass
```

### Phase 3: Person Detection and Tracking

#### Technologies and Libraries
- MediaPipe Pose for body detection
- OpenCV for tracking algorithms
- NumPy for position calculations

#### Implementation Steps
1. Implement full-body detection using MediaPipe Pose
2. Create person tracking across video frames
3. Develop person identification to distinguish between two people
4. Implement occlusion handling (when covered by blankets)

### Phase 4: Position Analysis

#### Technologies and Libraries
- NumPy for mathematical calculations
- SQLite for data storage
- Pandas for data analysis
- Matplotlib for visualization

#### Implementation Steps
1. Calculate body position relative to bed sides
2. Implement time tracking for positions
3. Create database schema for position data
4. Develop basic reporting and visualization

#### Data Model for Position Analysis

```python
# Position data structure
position_data = {
    'timestamp': '2025-03-17T23:45:12',  # ISO format timestamp
    'person_id': 1,                       # 1 or 2 to identify person
    'bed_side': 'left',                   # 'left' or 'right'
    'position_percentage': {              # Percentage of body on each side
        'left': 0.75,                     # 75% on left side
        'right': 0.25                     # 25% on right side
    },
    'body_position': 'side',              # 'back', 'side', 'stomach', etc.
    'movement_level': 0.2                 # Movement intensity (0.0 to 1.0)
}
```

### Phase 5: Night Vision Integration

#### Technologies and Libraries
- OpenCV for camera integration and image processing
- Hardware-specific SDKs for night vision camera

#### Implementation Steps
1. Research and select appropriate night vision camera
2. Implement camera interface for the selected hardware
3. Optimize detection algorithms for low-light conditions
4. Test and refine the system with actual sleep scenarios

## Resources and References

- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [MediaPipe Face Detection Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python)
- [MediaPipe Pose Estimation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [NumPy Documentation](https://numpy.org/doc/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [Matplotlib Visualization Guide](https://matplotlib.org/stable/users/explain/quick_start.html)
