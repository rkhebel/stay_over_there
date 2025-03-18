"""
Sleep Position Analysis System - Face Detection Application

This application provides face detection functionality using a modular approach.
It integrates camera selection, camera interface, and face detection modules.

Usage:
1. Run the script: python structured_face_detection_app.py
2. Select a camera from the graphical interface
3. Press 'q' to quit the application
"""

import cv2
import time
import sys
import traceback
import numpy as np

from camera_selector import CameraSelector
from camera_module import CameraInterface
from face_detector import FaceDetector

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        import mediapipe
        print(f"✓ MediaPipe version: {mediapipe.__version__}")
        import numpy
        print(f"✓ NumPy version: {numpy.__version__}")
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install required packages using:\n")
        print("  pip install -r requirements.txt\n")
        return False

def check_environment():
    """Check if running in a virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("\nWARNING: Not running in a virtual environment. This may cause issues with dependencies.")
        print("Consider creating and activating a virtual environment as described in the README.\n")

def setup_camera():
    """Set up camera selection and initialization"""
    # Select camera
    camera_selector = CameraSelector()
    camera_index = camera_selector.select_camera()
    
    if camera_index is None:
        print("\n✗ Camera selection canceled.\n")
        return None
    
    print(f"\nSelected camera index: {camera_index}")
    
    # Initialize the selected camera
    print("Initializing camera...")
    camera = CameraInterface(camera_id=camera_index, width=640, height=480)
    if not camera.start():
        print("\n✗ ERROR: Could not open camera. Please check your webcam connection.\n")
        return None
    
    # Get actual camera properties
    properties = camera.get_camera_properties()
    print(f"Camera started with resolution: {properties['width']}x{properties['height']}, FPS: {properties['fps']}")
    
    return camera, properties

def run_face_detection(camera, properties):
    """Run the face detection loop"""
    # Initialize face detector
    print("Initializing face detector...")
    detector = FaceDetector(min_detection_confidence=0.5)
    
    # Variables for tracking performance
    frame_count = 0
    face_count_total = 0
    start_time = time.time()
    
    # Create window
    window_name = "Sleep Position Analysis - Face Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("\n✓ Setup complete! Camera feed starting...\n")
    print("Press 'q' to quit the application\n")
    
    try:
        while True:
            # Capture frame
            frame, fps = camera.get_frame()
            
            if frame is None:
                print(f"Failed to capture frame (attempt {frame_count+1})")
                # Only try to reinitialize if we've successfully captured frames before
                if frame_count > 0:
                    camera.stop()
                    print("Reinitializing camera...")
                    if not camera.start():
                        print("Failed to reinitialize camera")
                        break
                else:
                    print("Camera not providing frames. Please try another camera.")
                    break
                continue
            
            frame_count += 1
            
            # Process the frame with face detection
            processed_frame, face_data = detector.detect_faces(frame)
            
            # Calculate elapsed time and average FPS
            elapsed_time = time.time() - start_time
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Update face count total
            face_count_frame = len(face_data)
            face_count_total += face_count_frame
            
            # Add display information
            display_info(processed_frame, fps, properties, frame_count)
            
            # Show the frame
            cv2.imshow(window_name, processed_frame)
            
            # Break loop on 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User pressed 'q'. Exiting...")
                break
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        traceback.print_exc()
    
    finally:
        # Release resources
        camera.stop()
        cv2.destroyAllWindows()
        print("\n✓ Application closed successfully\n")

def display_info(frame, fps, properties, frame_count):
    """Display information on the frame"""
    # Add FPS information
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add resolution information
    cv2.putText(frame, f"Resolution: {properties['width']}x{properties['height']}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add frame count
    cv2.putText(frame, f"Frame count: {frame_count}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display quit instructions
    cv2.putText(frame, "Press 'q' to quit", 
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

def main():
    print("\n=== SLEEP POSITION ANALYSIS SYSTEM - FACE DETECTION ===\n")
    
    # Check dependencies and environment
    if not check_dependencies():
        return
    
    check_environment()
    
    # Set up camera
    camera_setup = setup_camera()
    if camera_setup is None:
        return
    
    camera, properties = camera_setup
    
    # Run face detection
    run_face_detection(camera, properties)

if __name__ == "__main__":
    main()
