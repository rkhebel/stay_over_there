import cv2
import mediapipe as mp
import time

def main():
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,  # 0 for short-range detection (within 2 meters)
        min_detection_confidence=0.5
    )

    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)  # 0 is usually the built-in webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam initialized successfully!")
    
    # Get webcam properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera resolution: {frame_width}x{frame_height}, FPS: {fps}")
    
    # Create window
    cv2.namedWindow("Face Detection Demo", cv2.WINDOW_NORMAL)
    
    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Face Detection
            results = face_detection.process(rgb_frame)
            
            # Draw face detections
            if results.detections:
                for detection in results.detections:
                    # Draw bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Display confidence score
                    confidence = round(detection.score[0] * 100)
                    cv2.putText(frame, f"{confidence}%", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Display the number of faces detected
            face_count = len(results.detections) if results.detections else 0
            cv2.putText(frame, f"Faces: {face_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display the resulting frame
            cv2.imshow("Face Detection Demo", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Release the webcam and close all windows
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")

if __name__ == "__main__":
    main()
