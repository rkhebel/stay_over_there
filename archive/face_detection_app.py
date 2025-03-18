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
