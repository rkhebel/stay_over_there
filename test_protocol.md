# Face Detection Testing Protocol

## Basic Functionality Tests

1. **Camera Initialization**
   - Run `test_camera.py`
   - Verify camera starts without errors
   - Confirm video feed displays properly
   - Check FPS counter is working

2. **Face Detection Accuracy**
   - Run `face_detection_app.py`
   - Test with single face at different distances (0.5m, 1m, 2m)
   - Test with multiple faces in frame
   - Verify bounding boxes are correctly placed around faces
   - Check confidence scores are displayed

3. **Performance Testing**
   - Monitor FPS during operation
   - Target: Maintain at least 15 FPS
   - Note any performance degradation patterns

## Environmental Tests

1. **Lighting Conditions**
   - Test in bright lighting
   - Test in moderate lighting
   - Test in low lighting
   - Note detection accuracy in each condition

2. **Angles and Occlusion**
   - Test with face at different angles (front, 45°, profile)
   - Test with partial face occlusion (hand, hair, glasses)
   - Note detection limitations

## Refinement Checklist

- [ ] Optimize for consistent FPS
- [ ] Adjust detection confidence threshold if needed
- [ ] Improve visualization (colors, text size, etc.)
- [ ] Add additional information display if useful
