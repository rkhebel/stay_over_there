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
