"""
Simple Camera Test with Graphical UI

This script provides a basic test for camera selection and display.
It allows you to select from available cameras using a graphical interface
and displays the video feed.

USAGE:
1. Run the script: python simple_camera_test.py
2. Click on a camera from the graphical selection menu
3. Press 'q' to quit the camera view
4. Press 'ESC' to quit the selection menu
"""

import cv2
import time
import sys
import numpy as np

def create_graphical_camera_selection(available_cameras):
    """Create a graphical UI for camera selection using OpenCV"""
    # Create a window for camera selection
    window_name = "Camera Selection"
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
        text = f"Camera {camera['index']}: {camera['width']}x{camera['height']} @ {int(camera['fps'])} FPS"
        cv2.putText(ui_image, text, (margin + 10, button_top + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Additional info
        info_text = f"Backend: {camera['name']} - Frame test: {camera['frame_test']}"
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

def main():
    print("\n=== SIMPLE CAMERA TEST WITH GRAPHICAL UI ===\n")
    
    # Scan for available cameras
    print("Scanning for available cameras...")
    available_cameras = []
    
    for i in range(10):  # Check first 10 camera indices
        try:
            print(f"Testing camera {i}...", end="")
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                backend = cap.getBackendName()
                
                # Try to read a test frame
                ret, frame = cap.read()
                frame_success = ret and frame is not None
                
                details = {
                    "index": i,
                    "name": backend,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frame_test": "Success" if frame_success else "Failed"
                }
                
                available_cameras.append(details)
                print(f" Found: {width}x{height} @ {fps} FPS - {backend} - Frame test: {details['frame_test']}")
            else:
                print(" Not available")
                
            cap.release()
        except Exception as e:
            print(f" Error: {e}")
    
    if not available_cameras:
        print("\nNo cameras found on the system")
        return
    
    # Display available cameras in console
    print("\nAvailable cameras:")
    for i, camera in enumerate(available_cameras):
        print(f"{i+1}. Camera {camera['index']}: {camera['width']}x{camera['height']} @ {camera['fps']} FPS - {camera['name']} - Frame test: {camera['frame_test']}")
    
    # Show graphical camera selection UI
    print("\nLaunching graphical camera selection interface...")
    camera_index = create_graphical_camera_selection(available_cameras)
    
    # If no camera was selected (user pressed ESC)
    if camera_index is None:
        print("Camera selection canceled")
        return
        
    print(f"Selected camera index: {camera_index}")
    
    # Initialize the selected camera
    print(f"\nInitializing camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera started with resolution: {width}x{height}, FPS: {fps}")
    
    # Create window
    window_name = "Camera Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    
    print("\nCamera feed starting...")
    print("Press 'q' to quit")
    
    # Variables for FPS calculation
    prev_time = 0
    frame_count = 0
    
    try:
        # Main loop
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print(f"Failed to capture frame (attempt {frame_count+1})")
                # Try to reinitialize camera after multiple failures
                if frame_count > 0:  # Only try to reinitialize if we've successfully captured at least one frame
                    cap.release()
                    print("Reinitializing camera...")
                    cap = cv2.VideoCapture(camera_index)
                    if not cap.isOpened():
                        print("Failed to reinitialize camera")
                        break
                else:
                    # If we've never captured a frame, just exit
                    print("Camera not providing frames. Please try another camera.")
                    break
                continue
            
            frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            if prev_time > 0:
                fps = 1 / (current_time - prev_time)
            else:
                fps = 0
            prev_time = current_time
            
            # Add FPS and resolution text
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Resolution: {width}x{height}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame count: {frame_count}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show the frame
            cv2.imshow(window_name, frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User pressed 'q'. Exiting...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during camera operation: {e}")
    finally:
        # Clean up
        print("Releasing camera resources...")
        cap.release()
        cv2.destroyAllWindows()
        print("Camera test completed")

if __name__ == "__main__":
    main()
