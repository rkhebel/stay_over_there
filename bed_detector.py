"""
Bed Detector Module

This module provides functionality for defining and detecting the bed area in a camera frame.
It allows users to define the bed boundaries and dividing line between sides.

Usage:
1. Initialize the BedDetector
2. Use the define_bed_area method to set up the bed boundaries
3. Use the detect_bed method to detect the bed in subsequent frames
"""

import cv2
import numpy as np
import json
import os

class BedDetector:
    def __init__(self, config_file="bed_config.json"):
        """
        Initialize the bed detector.
        
        Args:
            config_file: Path to the configuration file for bed boundaries
        """
        self.config_file = config_file
        self.bed_corners = []  # [top-left, top-right, bottom-right, bottom-left]
        self.dividing_line = []  # [top-point, bottom-point]
        self.is_configured = False
        
        # Try to load existing configuration
        self.load_configuration()
    
    def load_configuration(self):
        """
        Load bed configuration from file if it exists.
        
        Returns:
            bool: True if configuration was loaded successfully, False otherwise
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                self.bed_corners = config.get('bed_corners', [])
                self.dividing_line = config.get('dividing_line', [])
                
                if len(self.bed_corners) == 4 and len(self.dividing_line) == 2:
                    self.is_configured = True
                    print(f"Bed configuration loaded from {self.config_file}")
                    return True
                    
            except Exception as e:
                print(f"Error loading bed configuration: {e}")
        
        return False
    
    def save_configuration(self):
        """
        Save bed configuration to file.
        
        Returns:
            bool: True if configuration was saved successfully, False otherwise
        """
        if not self.is_configured:
            print("Bed is not configured yet")
            return False
            
        config = {
            'bed_corners': self.bed_corners,
            'dividing_line': self.dividing_line
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
            print(f"Bed configuration saved to {self.config_file}")
            return True
        except Exception as e:
            print(f"Error saving bed configuration: {e}")
            return False
    
    def define_bed_area(self, frame):
        """
        Interactive method to define the bed area and dividing line.
        
        Args:
            frame: The camera frame to use for defining the bed area
            
        Returns:
            bool: True if bed area was defined successfully, False otherwise
        """
        # Make a copy of the frame to draw on
        display_frame = frame.copy()
        
        # Instructions
        cv2.putText(display_frame, "Define bed corners: Click on the 4 corners in order:", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "1. Top-left, 2. Top-right, 3. Bottom-right, 4. Bottom-left", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'r' to reset, 'c' to continue to dividing line, 'q' to quit", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Create a window and set mouse callback
        window_name = "Define Bed Area"
        cv2.namedWindow(window_name)
        
        # Clear previous configuration
        self.bed_corners = []
        self.dividing_line = []
        self.is_configured = False
        
        # Mouse callback function
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.bed_corners) < 4:
                    self.bed_corners.append((x, y))
                    # Draw the point
                    cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                    # Draw the bed outline as it's being defined
                    if len(self.bed_corners) > 1:
                        for i in range(len(self.bed_corners) - 1):
                            cv2.line(display_frame, self.bed_corners[i], self.bed_corners[i+1], (0, 255, 0), 2)
                    # If we have 4 points, connect the last to the first
                    if len(self.bed_corners) == 4:
                        cv2.line(display_frame, self.bed_corners[3], self.bed_corners[0], (0, 255, 0), 2)
                        cv2.putText(display_frame, "Press 'c' to continue to dividing line", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # Main loop for bed corners definition
        while True:
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset
                display_frame = frame.copy()
                cv2.putText(display_frame, "Define bed corners: Click on the 4 corners in order:", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "1. Top-left, 2. Top-right, 3. Bottom-right, 4. Bottom-left", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'r' to reset, 'c' to continue to dividing line, 'q' to quit", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.bed_corners = []
            
            elif key == ord('c') and len(self.bed_corners) == 4:  # Continue to dividing line
                break
                
            elif key == ord('q'):  # Quit
                cv2.destroyWindow(window_name)
                return False
        
        # Now define the dividing line
        display_frame = frame.copy()
        
        # Draw the bed outline
        for i in range(4):
            cv2.line(display_frame, self.bed_corners[i], self.bed_corners[(i+1)%4], (0, 255, 0), 2)
            
        # Instructions for dividing line
        cv2.putText(display_frame, "Define dividing line: Click on 2 points to divide the bed:", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "1. Top point, 2. Bottom point", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'r' to reset, 's' to save, 'q' to quit", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update mouse callback for dividing line
        def dividing_line_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.dividing_line) < 2:
                    self.dividing_line.append((x, y))
                    # Draw the point
                    cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                    # Draw the dividing line as it's being defined
                    if len(self.dividing_line) == 2:
                        cv2.line(display_frame, self.dividing_line[0], self.dividing_line[1], (255, 0, 0), 2)
                        cv2.putText(display_frame, "Press 's' to save configuration", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.setMouseCallback(window_name, dividing_line_callback)
        
        # Main loop for dividing line definition
        while True:
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset dividing line
                display_frame = frame.copy()
                # Redraw the bed outline
                for i in range(4):
                    cv2.line(display_frame, self.bed_corners[i], self.bed_corners[(i+1)%4], (0, 255, 0), 2)
                
                cv2.putText(display_frame, "Define dividing line: Click on 2 points to divide the bed:", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "1. Top point, 2. Bottom point", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'r' to reset, 's' to save, 'q' to quit", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.dividing_line = []
            
            elif key == ord('s') and len(self.dividing_line) == 2:  # Save configuration
                self.is_configured = True
                self.save_configuration()
                break
                
            elif key == ord('q'):  # Quit
                cv2.destroyWindow(window_name)
                return False
        
        cv2.destroyWindow(window_name)
        return True
    
    def detect_bed(self, frame):
        """
        Detect the bed in the frame based on the defined bed area.
        
        Args:
            frame: The camera frame to detect the bed in
            
        Returns:
            tuple: (processed_frame, bed_data)
                processed_frame: Frame with bed visualization
                bed_data: Dictionary with bed information
        """
        if not self.is_configured:
            return frame, None
            
        # Make a copy of the frame to draw on
        processed_frame = frame.copy()
        
        # Draw the bed outline
        for i in range(4):
            cv2.line(processed_frame, self.bed_corners[i], self.bed_corners[(i+1)%4], (0, 255, 0), 2)
            
        # Draw the dividing line
        cv2.line(processed_frame, self.dividing_line[0], self.dividing_line[1], (255, 0, 0), 2)
        
        # Label the sides
        left_center = ((self.bed_corners[0][0] + self.dividing_line[0][0]) // 2, 
                      (self.bed_corners[0][1] + self.dividing_line[0][1]) // 2)
        right_center = ((self.bed_corners[1][0] + self.dividing_line[0][0]) // 2, 
                       (self.bed_corners[1][1] + self.dividing_line[0][1]) // 2)
        
        cv2.putText(processed_frame, "Left Side", left_center, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_frame, "Right Side", right_center, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Create a mask for the bed area
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        bed_corners_array = np.array(self.bed_corners, dtype=np.int32)
        cv2.fillPoly(mask, [bed_corners_array], 255)
        
        # Calculate bed data
        bed_data = {
            'corners': self.bed_corners,
            'dividing_line': self.dividing_line,
            'mask': mask
        }
        
        return processed_frame, bed_data
    
    def is_point_in_bed(self, point):
        """
        Check if a point is inside the bed area.
        
        Args:
            point: (x, y) coordinates to check
            
        Returns:
            bool: True if point is in bed area, False otherwise
        """
        if not self.is_configured:
            return False
            
        # Convert bed corners to numpy array
        bed_corners_array = np.array(self.bed_corners, dtype=np.int32)
        
        # Check if point is inside the polygon
        result = cv2.pointPolygonTest(bed_corners_array, point, False)
        return result >= 0
    
    def get_side(self, point):
        """
        Determine which side of the bed a point is on.
        
        Args:
            point: (x, y) coordinates to check
            
        Returns:
            str: 'left', 'right', or None if point is not in bed
        """
        if not self.is_configured or not self.is_point_in_bed(point):
            return None
            
        # Calculate the side based on the dividing line
        # We'll use the cross product to determine which side of the line the point is on
        x, y = point
        x1, y1 = self.dividing_line[0]
        x2, y2 = self.dividing_line[1]
        
        # Calculate the cross product
        cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        
        # Determine the side
        if cross_product < 0:
            return 'left'
        else:
            return 'right'
    
    def calculate_position(self, body_points):
        """
        Calculate the position of a person relative to the bed sides.
        
        Args:
            body_points: List of (x, y) coordinates representing body points
            
        Returns:
            dict: Position data including side percentages
        """
        if not self.is_configured or not body_points:
            return None
            
        # Count points on each side
        left_count = 0
        right_count = 0
        
        for point in body_points:
            side = self.get_side(point)
            if side == 'left':
                left_count += 1
            elif side == 'right':
                right_count += 1
        
        total_points = left_count + right_count
        if total_points == 0:
            return None
            
        # Calculate percentages
        left_percentage = left_count / total_points
        right_percentage = right_count / total_points
        
        # Determine primary side
        primary_side = 'left' if left_percentage >= right_percentage else 'right'
        
        position_data = {
            'left_percentage': left_percentage,
            'right_percentage': right_percentage,
            'primary_side': primary_side
        }
        
        return position_data


def main():
    """
    Main function to test the bed detector.
    """
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize bed detector
    bed_detector = BedDetector()
    
    # Check if bed is already configured
    if not bed_detector.is_configured:
        print("Bed not configured. Let's define the bed area.")
        
        # Capture a frame for bed definition
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            cap.release()
            return
        
        # Define bed area
        if not bed_detector.define_bed_area(frame):
            print("Bed definition canceled")
            cap.release()
            return
    
    print("Bed configured. Press 'q' to quit, 'r' to reconfigure bed.")
    
    # Main loop
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect bed
        processed_frame, bed_data = bed_detector.detect_bed(frame)
        
        # Show the frame
        cv2.imshow("Bed Detection", processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reconfigure bed
            if bed_detector.define_bed_area(frame):
                print("Bed reconfigured")
            else:
                print("Bed reconfiguration canceled")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
