import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        """
        Initialize the MediaPipe face detector.
        
        Args:
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
                detection to be considered successful. (default: 0.5)
        """
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range detection (within 2 meters)
            min_detection_confidence=min_detection_confidence
        )
        
    def detect_faces(self, frame):
        """
        Detect faces in the given frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            processed_frame: Frame with detection visualizations
            face_data: List of dictionaries containing face detection data
                Each dict contains: 'bbox' (x, y, w, h), 'score', 'landmarks'
        """
        if frame is None:
            return None, []
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Face Detection
        results = self.face_detection.process(rgb_frame)
        
        # Create a copy of the frame for drawing
        processed_frame = frame.copy()
        
        # List to store face data
        face_data = []
        
        # Check if any faces were detected
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox_rel = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bbox_rel.xmin * iw), int(bbox_rel.ymin * ih), \
                            int(bbox_rel.width * iw), int(bbox_rel.height * ih)
                
                # Draw bounding box
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Get confidence score
                score = detection.score[0]
                
                # Display confidence score
                confidence_text = f"{int(score * 100)}%"
                cv2.putText(processed_frame, confidence_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Extract face landmarks
                landmarks = []
                for i in range(6):  # MediaPipe provides 6 landmarks for each face
                    try:
                        landmark = detection.location_data.relative_keypoints[i]
                        lx, ly = int(landmark.x * iw), int(landmark.y * ih)
                        landmarks.append((lx, ly))
                        
                        # Draw landmark points
                        cv2.circle(processed_frame, (lx, ly), 5, (255, 0, 0), -1)
                    except IndexError:
                        continue
                
                # Store face data
                face_data.append({
                    'bbox': (x, y, w, h),
                    'score': score,
                    'landmarks': landmarks
                })
        
        # Display face count
        face_count = len(face_data)
        cv2.putText(processed_frame, f"Faces: {face_count}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return processed_frame, face_data
