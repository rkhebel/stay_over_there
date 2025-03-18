# Sleep Position Analysis System - Technical Specifications

## 1. Camera Requirements & Options

Based on our research, we have several options for the night vision camera component:

### Option 1: Security Camera with Starlight/Low-Light Sensor
- **Pros**: 
  - Excellent low-light performance without IR illumination
  - Higher resolution (up to 4K)
  - Weatherproof (though not needed for indoor use)
  - Often includes onboard storage options
- **Cons**:
  - Typically more expensive
  - May need to be integrated with a network video recorder (NVR)
  - Often requires ethernet connectivity
  - May be overkill for bedroom installation
- **Recommended Models**:
  - Annke 4K Bullet Security Camera (Minimum illumination: 0.001 lux)
  - Reolink RLC-810A (with color night vision)

### Option 2: USB Night Vision Webcam
- **Pros**:
  - Direct USB connection to computer
  - Easier to integrate with custom software
  - More discreet for bedroom installation
  - Generally less expensive
- **Cons**:
  - Typically lower resolution
  - Shorter IR illumination range
  - Less sophisticated night vision capabilities
- **Recommended Models**:
  - Logitech C922 Pro Stream (with enhanced low-light performance)
  - Arducam 1080P USB Camera with Day & Night Vision (with automatic IR-cut switching)

### Option 3: Dedicated IR Camera Module 
- **Pros**:
  - Highly customizable
  - Can be integrated with Raspberry Pi or similar
  - Often more compact
- **Cons**: 
  - Requires more technical setup
  - May need additional IR illuminators
- **Recommended Models**:
  - Raspberry Pi NoIR Camera Module
  - ESP32-CAM with IR LEDs

### Recommended Configuration
Based on the project requirements, we recommend:
- A USB night vision webcam for initial development and testing
- Ceiling-mounted position for optimal bed coverage
- Separate, gentle IR illuminator to enhance visibility without disturbing sleep
- Consider upgrading to a higher-end security camera for final implementation if better resolution/detection is needed

## 2. System Architecture

### 2.1. Hardware Components
- **Camera**: USB night vision webcam or IP camera with IR capabilities
- **Processing Unit**: Computer (Mac initially, potentially Raspberry Pi for final deployment)
- **Optional IR Illuminator**: Low-intensity IR illuminator to enhance night vision
- **Camera Mount**: Ceiling mount for optimal bed coverage

### 2.2. Software Components

#### 2.2.1. Camera Interface Layer
- **Purpose**: Interface with the camera device, handle video capture
- **Technologies**: 
  - OpenCV's VideoCapture API
  - Camera-specific SDKs if needed
- **Key Functions**:
  - Initialize camera with appropriate settings
  - Capture frames at defined intervals (1-5 FPS should be sufficient for sleep tracking)
  - Apply any necessary image preprocessing
  - Check for camera disconnections and recovery

#### 2.2.2. Bed Mapping Module
- **Purpose**: Define the bed area and the dividing line between sides
- **Technologies**:
  - OpenCV for image processing
  - Custom GUI for boundary definition
- **Key Functions**:
  - Allow user to define bed corners in the camera frame
  - Define the central dividing line between "sides"
  - Save mapping configuration for reuse
  - Provide recalibration capability if camera moves
  - Transform perspective to get a top-down view if camera is at an angle

#### 2.2.3. Person Detection and Tracking Module
- **Purpose**: Identify and track people in the bed
- **Technologies**:
  - MediaPipe for pose estimation
  - YOLO for person detection
  - OpenCV for image processing
- **Key Functions**:
  - Detect human figures in bed
  - Differentiate between two people
  - Handle occlusion (blankets, pillows)
  - Track position changes over time
  - Estimate body boundaries for position analysis

#### 2.2.4. Position Analysis Module
- **Purpose**: Analyze the positions relative to bed sides
- **Technologies**:
  - NumPy for mathematical operations
  - Custom algorithms for position analysis
- **Key Functions**:
  - Calculate percentage of each person on each side
  - Track position over time
  - Handle edge cases (sitting up, getting in/out of bed)
  - Generate position snapshots at regular intervals

#### 2.2.5. Data Storage Module
- **Purpose**: Store position data securely
- **Technologies**:
  - SQLite for local database
  - JSON for configuration files
- **Key Functions**:
  - Store timestamped position data
  - Avoid storing actual images for privacy
  - Implement data retention policies
  - Provide data export capabilities

#### 2.2.6. Reporting and Visualization Module
- **Purpose**: Generate insights from position data
- **Technologies**:
  - Matplotlib/Seaborn for visualizations
  - Pandas for data analysis
  - Flask for web dashboard (optional)
- **Key Functions**:
  - Calculate time spent on each side
  - Generate heatmaps of positions
  - Provide nightly and trend reports
  - Visualize movement patterns

## 3. Implementation Plan

### Phase 1: Setup and Camera Integration (1-2 weeks)
- Select and purchase appropriate camera
- Set up development environment
- Develop camera interface module
- Test basic video capture in both daylight and night conditions

### Phase 2: Bed Mapping (1 week)
- Develop interface for defining bed boundaries
- Implement bed side division logic
- Test boundary detection with sample videos

### Phase 3: Person Detection (2 weeks)
- Implement person detection model
- Test in various lighting conditions
- Develop person differentiation logic
- Handle occlusion scenarios

### Phase 4: Position Analysis (2 weeks)
- Develop algorithms for position tracking
- Implement side-based analysis
- Create time-based position tracking
- Test with sample sleep recordings

### Phase 5: Data Storage and Reporting (1 week)
- Implement database schema
- Create position data storage logic
- Develop basic reporting functions
- Ensure privacy considerations

### Phase 6: Testing and Refinement (2 weeks)
- Deploy for actual sleep monitoring
- Gather feedback on accuracy
- Refine detection and analysis algorithms
- Optimize performance

## 4. Privacy Considerations

### 4.1. Data Storage
- Raw video footage will NOT be stored
- Only abstract position data will be saved
- No facial or identifying features will be included in stored data

### 4.2. Processing
- All processing will happen locally on the device
- No cloud services will be used for sensitive data processing
- Option to blur/abstract the video feed during testing

### 4.3. Security
- Password protection for the application
- Encryption for any stored data
- Option to automatically disable recording when specified conditions are met

## 5. Next Steps

1. **Select a camera** from the recommended options
2. **Set up the development environment** with required libraries
3. **Create a basic proof-of-concept** that captures and displays video from the camera
4. **Develop the bed mapping interface** to define the monitoring area
5. **Test person detection algorithms** in a bedroom setting under night conditions

## 6. Required Libraries and Tools

- **Python 3.8+**: Core programming language
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Human pose estimation
- **NumPy**: Numerical processing
- **SQLite**: Local data storage
- **Pandas**: Data analysis
- **Matplotlib/Seaborn**: Data visualization
- **Flask**: Web dashboard (optional)
