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
import numpy as np
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

def list_camera_properties(camera_index):
    """List all available properties of a camera"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}")
        return
    
    print(f"\nDetailed properties for camera {camera_index}:")
    print(f"Backend: {cap.getBackendName()}")
    
    # Common properties to check
    properties = [
        (cv2.CAP_PROP_FRAME_WIDTH, "Width"),
        (cv2.CAP_PROP_FRAME_HEIGHT, "Height"),
        (cv2.CAP_PROP_FPS, "FPS"),
        (cv2.CAP_PROP_BRIGHTNESS, "Brightness"),
        (cv2.CAP_PROP_CONTRAST, "Contrast"),
        (cv2.CAP_PROP_SATURATION, "Saturation"),
        (cv2.CAP_PROP_HUE, "Hue"),
        (cv2.CAP_PROP_GAIN, "Gain"),
        (cv2.CAP_PROP_EXPOSURE, "Exposure"),
        (cv2.CAP_PROP_CONVERT_RGB, "Convert RGB"),
        (cv2.CAP_PROP_BUFFERSIZE, "Buffer Size")
    ]
    
    for prop_id, prop_name in properties:
        value = cap.get(prop_id)
        print(f"{prop_name}: {value}")
    
    # Release the camera
    cap.release()

def create_graphical_camera_selection(available_cameras):
    """Create a graphical UI for camera selection using OpenCV"""
    # Create a window for camera selection
    window_name = "Camera Selection - Sleep Position Analysis"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    # Calculate UI dimensions
    margin = 20
    button_height = 60
    button_width = 760
    button_spacing = 20
    
    # Total height needed for all cameras
    total_height = margin + len(available_cameras) * (button_height + button_spacing) + margin
    
    # Create a blank image for the UI (light gray background)
    ui_image = np.ones((max(600, total_height), 800, 3), dtype=np.uint8) * 240
    
    # Add title
    cv2.putText(ui_image, "Select a Camera", (margin, margin + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add instructions
    cv2.putText(ui_image, "Click on a camera to select it or press ESC to quit", 
                (margin, margin + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Define button areas
    button_areas = []
    
    # Draw camera options as buttons
    y_pos = margin + 80
    for i, camera in enumerate(available_cameras):
        # Button background
        button_top = y_pos
        button_bottom = y_pos + button_height
        
        # Store button area for click detection
        button_areas.append((margin, button_top, margin + button_width, button_bottom, camera["index"]))
        
        # Draw button
        cv2.rectangle(ui_image, (margin, button_top), (margin + button_width, button_bottom), 
                     (200, 200, 200), -1)  # Fill
        cv2.rectangle(ui_image, (margin, button_top), (margin + button_width, button_bottom), 
                     (100, 100, 100), 2)   # Border
        
        # Button text
        iphone_indicator = " (Likely iPhone)" if camera.get("likely_iphone", False) else ""
        text = f"Camera {camera['index']}: {camera['width']}x{camera['height']} @ {int(camera['fps'])} FPS{iphone_indicator}"
        cv2.putText(ui_image, text, (margin + 10, button_top + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Additional info
        info_text = f"Backend: {camera['name']} - Test: {camera.get('test_frame_success', 'Unknown')}"
        cv2.putText(ui_image, info_text, (margin + 10, button_top + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        
        # Update y position for next button
        y_pos += button_height + button_spacing
    
    # Show the selection UI and wait for mouse click
    selected_camera = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_camera
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is within any button area
            for left, top, right, bottom, camera_idx in button_areas:
                if left <= x <= right and top <= y <= bottom:
                    selected_camera = camera_idx
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Display UI until a camera is selected or ESC is pressed
    while selected_camera is None:
        cv2.imshow(window_name, ui_image)
        key = cv2.waitKey(100) & 0xFF
        if key == 27:  # ESC key
            break
    
    cv2.destroyWindow(window_name)
    return selected_camera

def create_camera_selection_ui():
    """Create a simple UI for camera selection using OpenCV"""
    print("\n===== Camera Selection Process Starting =====\n")
    print("Scanning for available cameras...")
    
    # Get available cameras
    available_cameras = []
    for i in range(10):  # Check first 10 camera indices
        try:
            print(f"Testing camera {i}...", end="")
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera properties
                details = {
                    "index": i,
                    "name": cap.getBackendName(),
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": cap.get(cv2.CAP_PROP_FPS)
                }
                
                # Try to determine if this might be an iPhone camera
                if details["name"] == "AVFOUNDATION" and details["fps"] > 25:
                    details["likely_iphone"] = True
                else:
                    details["likely_iphone"] = False
                    
                # Try to read a test frame
                ret, frame = cap.read()
                details["test_frame_success"] = "Success" if ret else "Failed"
                    
                print(f" Found: {details['width']}x{details['height']} @ {details['fps']} FPS" + 
                      (" (Likely iPhone)" if details["likely_iphone"] else ""))
                
                available_cameras.append(details)
            else:
                print(" Not available")
                
            cap.release()
        except Exception as e:
            print(f" Error: {e}")
    
    if not available_cameras:
        print("\n✗ ERROR: No cameras found on the system\n")
        return None
    
    # Print camera options in terminal for reference
    print("\nAvailable cameras:")
    for i, camera in enumerate(available_cameras):
        iphone_indicator = " (Likely iPhone)" if camera.get("likely_iphone", False) else ""
        print(f"{i+1}. Camera {camera['index']}: {camera['width']}x{camera['height']} @ {int(camera['fps'])} FPS{iphone_indicator}")
    
    # Show graphical camera selection UI
    print("\nLaunching graphical camera selection interface...")
    return create_graphical_camera_selection(available_cameras)

def main():
    print("\n===== DEBUG: Starting main function =====\n")
    
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
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range detection (within 2 meters)
            min_detection_confidence=0.5  # Minimum confidence threshold (0.0-1.0)
        )
        print("DEBUG: Face detector initialized successfully")
    except Exception as e:
        print(f"ERROR initializing face detector: {e}")
        return
    
    # Display camera selection UI
    print("\nLaunching camera selection interface...")
    camera_index = create_camera_selection_ui()
    
    if camera_index is None:
        print("\n✗ Camera selection canceled by user.\n")
        return
    
    print(f"\nSelected camera index: {camera_index}")
    
    # Initialize the selected camera
    print("\n===== DEBUG: Initializing selected camera =====\n")
    print(f"DEBUG: Attempting to open camera with index {camera_index}")
    try:
        cap = cv2.VideoCapture(camera_index)
        print(f"DEBUG: VideoCapture object created for camera {camera_index}")
    except Exception as e:
        print(f"ERROR creating VideoCapture: {e}")
        return
    
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
    
    # Variables for tracking performance
    frame_count = 0
    face_count_total = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print(f"Failed to capture frame (attempt {frame_count+1})")
                # Only try to reinitialize if we've successfully captured frames before
                if frame_count > 0:
                    cap.release()
                    print("Reinitializing camera...")
                    cap = cv2.VideoCapture(camera_index)
                    if not cap.isOpened():
                        print("Failed to reinitialize camera")
                        break
                else:
                    print("Camera not providing frames. Please try another camera.")
                    break
                continue
            
            frame_count += 1
            
            # Calculate FPS
            curr_frame_time = time.time()
            fps = 1 / (curr_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = curr_frame_time
            
            # Create a copy of the frame for processing
            display_frame = frame.copy()
            
            # Convert to RGB for MediaPipe (MediaPipe uses RGB, OpenCV uses BGR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # To improve performance, mark the frame as not writeable to pass by reference
            rgb_frame.flags.writeable = False
            
            # Process the frame with MediaPipe face detection
            results = face_detection.process(rgb_frame)
            
            # Mark the frame as writeable again
            rgb_frame.flags.writeable = True
            
            # Draw face detections
            face_count_frame = 0
            if results.detections:
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
                    
                    face_count_frame += 1
                
                face_count_total += face_count_frame
            
            # Calculate elapsed time and average FPS
            elapsed_time = time.time() - start_time
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Display statistics
            cv2.putText(display_frame, f"Faces: {face_count_frame}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Resolution: {actual_width}x{actual_height}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frame count: {frame_count}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display quit instructions
            cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Show the frame
            cv2.imshow(window_name, display_frame)
            
            # Break loop on 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User pressed 'q'. Exiting...")
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
