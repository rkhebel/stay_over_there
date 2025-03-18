# Sleep Position Analysis System

## Project Overview

This project aims to develop a sleep position analysis system that tracks and analyzes how much time users spend on their designated sides of the bed. The system uses computer vision to detect and track people's positions during sleep.

## Current Status

We have implemented Phase 1 of the project: a minimal face detection demo using a webcam. This serves as a proof-of-concept for the person detection component that will be expanded in later phases.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam (built-in or external)
- Virtual environment (recommended)

### Installation

1. Clone this repository or download the files

2. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate on macOS/Linux
   source venv/bin/activate
   
   # Activate on Windows
   venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Demo

To run the minimal face detection demo:

```bash
python minimal_face_detection.py
```

Press 'q' to quit the application.

## Implementation Plan

The project is being developed in phases:

1. **Phase 1: Face Detection Demo (Current)** - Basic proof-of-concept using webcam
2. **Phase 2: Bed Area Mapping** - Define bed boundaries and sides
3. **Phase 3: Person Detection and Tracking** - Full-body detection and tracking
4. **Phase 4: Position Analysis** - Calculate and track positions relative to bed sides
5. **Phase 5: Night Vision Integration** - Adapt for low-light conditions

See `implementation_plan.md` for detailed specifications.

## Project Structure

- `minimal_face_detection.py` - Simplified face detection demo
- `implementation_plan.md` - Detailed project implementation plan
- `requirements.txt` - Python dependencies
- `test_protocol.md` - Testing guidelines

## License

This project is for personal use only.
