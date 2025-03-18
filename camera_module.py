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
