"""
Minimal Face Detection Demo - v1

This script provides a basic implementation of face detection using a webcam.
It uses OpenCV for camera access and MediaPipe for face detection.

REQUIREMENTS:
- Python 3.8+
- OpenCV (pip install opencv-python)
- MediaPipe (pip install mediapipe)

USAGE:
1. Install required packages in a virtual environment:
   python -m venv venv
   source venv/bin/activate (Mac/Linux) or venv\Scripts\activate (Windows)
   pip install -r requirements.txt

2. Run the script:
   python minimal_face_detection.py

3. Press 'q' to quit the application
"""

import cv2
import mediapipe as mp
import time
import sys

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        import mediapipe
        print(f"✓ MediaPipe version: {mediapipe.__version__}")
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install required packages using:\n")
        print("  pip install opencv-python mediapipe\n")
        return False

def main():
    # Check dependencies first
    if not check_dependencies():
        return
        
    # Check if running in a virtual environment
    import sys
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("\nWARNING: Not running in a virtual environment. This may cause issues with dependencies.")
        print("Consider creating and activating a virtual environment as described in the README.\n")
    
    print("\n=== SLEEP POSITION ANALYSIS SYSTEM - FACE DETECTION DEMO ===\n")
    
    # Initialize MediaPipe Face Detection
    print("Initializing face detector...")
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,  # 0 for short-range detection (within 2 meters)
        min_detection_confidence=0.5  # Minimum confidence threshold (0.0-1.0)
    )
    
    # Initialize webcam
    print("Starting camera...")
    
    # Try to list available cameras first
    print("DEBUG: Checking available cameras...")
    available_cameras = []
    for i in range(10):  # Check first 10 camera indices
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            print(f"DEBUG: Camera found at index {i}")
            available_cameras.append(i)
            temp_cap.release()
    
    if not available_cameras:
        print("\n✗ ERROR: No cameras found on the system\n")
        return
        
    # Try camera index 1 first (if available), then fall back to 0
    camera_index = 1 if 1 in available_cameras else available_cameras[0]
    print(f"DEBUG: Attempting to open camera with index {camera_index}")
    cap = cv2.VideoCapture(camera_index)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("\n✗ ERROR: Could not open camera. Please check your webcam connection.\n")
        print("DEBUG: Trying to list available cameras...")
        # Try to list available cameras
        for i in range(10):  # Check first 10 camera indices
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                print(f"DEBUG: Camera found at index {i}")
                temp_cap.release()
        return
    
    # Set resolution to 640x480 for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Get actual camera properties (may differ from requested)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera started with resolution: {actual_width}x{actual_height}, FPS: {fps}")
    print(f"DEBUG: Camera backend being used: {cap.getBackendName()}")
    
    # Variables for FPS calculation
    prev_frame_time = 0
    curr_frame_time = 0
    
    # Create window
    window_name = "Face Detection - Minimal Demo"
    print("DEBUG: Creating OpenCV window")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("DEBUG: Window created successfully")
    
    print("\n✓ Setup complete! Camera feed starting...\n")
    print("Press 'q' to quit the application\n")
    
    try:
        while True:
            # Capture frame
            print("DEBUG: Attempting to read frame", end="\r")
            ret, frame = cap.read()
            
            if not ret:
                print("\n✗ ERROR: Failed to capture frame from camera")
                print(f"DEBUG: Camera is still open: {cap.isOpened()}")
                
                # Try to reinitialize the camera
                print("DEBUG: Attempting to reinitialize camera...")
                cap.release()
                cap = cv2.VideoCapture(camera_index)
                if not cap.isOpened():
                    print("DEBUG: Failed to reinitialize camera")
                    break
                    
                print("DEBUG: Camera reinitialized, trying to read frame again")
                ret, frame = cap.read()
                if not ret:
                    print("DEBUG: Still failed to capture frame after reinitializing camera")
                    break
            
            print("DEBUG: Frame captured successfully", end="\r")
            
            # Calculate FPS
            curr_frame_time = time.time()
            fps = 1 / (curr_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = curr_frame_time
            
            # Create a copy of the frame for processing
            display_frame = frame.copy()
            
            # Convert to RGB for MediaPipe (MediaPipe uses RGB, OpenCV uses BGR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe face detection
            print("DEBUG: Processing frame with MediaPipe", end="\r")
            results = face_detection.process(rgb_frame)
            print("DEBUG: MediaPipe processing complete ", end="\r")
            
            # Draw face detections
            face_count = 0
            if results.detections:
                print(f"DEBUG: Found {len(results.detections)} faces", end="\r")
                for detection in results.detections:
                    # Get bounding box coordinates
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    
                    # Convert relative coordinates to absolute pixel values
                    x = max(0, int(bbox.xmin * iw))
                    y = max(0, int(bbox.ymin * ih))
                    w = min(int(bbox.width * iw), iw - x)
                    h = min(int(bbox.height * ih), ih - y)
                    
                    # Draw bounding box (green rectangle)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Get confidence score
                    score = detection.score[0]
                    
                    # Display confidence score above the bounding box
                    confidence_text = f"{int(score * 100)}%"
                    cv2.putText(display_frame, confidence_text, (x, max(y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Extract and display face landmarks (eyes, nose, mouth)
                    for i in range(6):  # MediaPipe provides 6 landmarks for each face
                        try:
                            # Get landmark coordinates
                            landmark = detection.location_data.relative_keypoints[i]
                            lx, ly = int(landmark.x * iw), int(landmark.y * ih)
                            
                            # Draw landmark points (blue circles)
                            cv2.circle(display_frame, (lx, ly), 3, (255, 0, 0), -1)
                        except IndexError:
                            continue
                    
                    face_count += 1
            
            # Display face count and FPS
            cv2.putText(display_frame, f"Faces: {face_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display quit instructions
            cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Show the frame
            print("DEBUG: Displaying frame", end="\r")
            cv2.imshow(window_name, display_frame)
            
            # Break loop on 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nDEBUG: 'q' key pressed, exiting loop")
                break
    
    except Exception as e:
        import traceback
        print(f"\n✗ ERROR: {e}\n")
        print("DEBUG: Full exception traceback:")
        traceback.print_exc()
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Application closed successfully\n")

if __name__ == "__main__":
    main()
