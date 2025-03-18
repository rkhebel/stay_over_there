"""
Camera Selection Module

This module provides functionality for detecting and selecting available cameras
using a graphical user interface.
"""

import cv2
import numpy as np

class CameraSelector:
    def __init__(self):
        """Initialize the camera selector."""
        self.available_cameras = []
    
    def scan_for_cameras(self, max_cameras=10):
        """
        Scan for available cameras on the system.
        
        Args:
            max_cameras: Maximum number of camera indices to check
            
        Returns:
            List of dictionaries containing camera details
        """
        print("\n===== Camera Selection Process Starting =====\n")
        print("Scanning for available cameras...")
        
        available_cameras = []
        for i in range(max_cameras):
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
            return []
        
        # Print camera options in terminal for reference
        print("\nAvailable cameras:")
        for i, camera in enumerate(available_cameras):
            iphone_indicator = " (Likely iPhone)" if camera.get("likely_iphone", False) else ""
            print(f"{i+1}. Camera {camera['index']}: {camera['width']}x{camera['height']} @ {int(camera['fps'])} FPS{iphone_indicator}")
        
        self.available_cameras = available_cameras
        return available_cameras
    
    def create_graphical_camera_selection(self):
        """
        Create a graphical UI for camera selection using OpenCV.
        
        Returns:
            int: Selected camera index or None if selection was canceled
        """
        if not self.available_cameras:
            print("No cameras available for selection")
            return None
            
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
        total_height = margin + len(self.available_cameras) * (button_height + button_spacing) + margin
        
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
        for i, camera in enumerate(self.available_cameras):
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
    
    def select_camera(self):
        """
        Scan for available cameras and show selection UI.
        
        Returns:
            int: Selected camera index or None if selection was canceled
        """
        self.scan_for_cameras()
        
        if not self.available_cameras:
            return None
            
        print("\nLaunching graphical camera selection interface...")
        return self.create_graphical_camera_selection()
