# Parking Slot Detection and Counter

Welcome to the Parking Slot Detection and Counter project! This project is designed to detect and count the number of empty parking slots in a given parking lot using video footage and a machine learning model. The project leverages computer vision techniques and a Support Vector Classifier (SVC) model to classify parking slots as either empty or occupied.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Explanation of Files](#explanation-of-files)
- [Acknowledgements](#acknowledgements)

## Overview
The goal of this project is to provide a real-time count of available parking spots in a parking lot using a video feed. The project processes each frame of the video to identify parking slots and determine their occupancy status.

## Project Structure
.\
├── input_data/\
│ ├── mask_1920_1080.png\
│ ├── parking_1920_1080_loop.mp4\
├── model.py\
├── utils.py\
├── parking_classifier/\
│ ├── model.p\
├── README.md


## Requirements
- Python 3.x
- OpenCV
- NumPy
- scikit-image
- scikit-learn
- Matplotlib

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/parking-slot-detection.git
    cd parking-slot-detection
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Ensure the model file `model.p` is located in the `parking_classifier` directory.

## Usage
1. Get the required dataset from this link [Input Data](https://drive.google.com/file/d/1ZLVfsBQ7RKPjjrXnUdzviaB2qDpmuTYx/view?usp=sharing)
1. Place the video file (`parking_1920_1080_loop.mp4`) and the mask file (`mask_1920_1080.png`) in the `input_data` directory.

2. Run the `model.py` script:
    ```sh
    python model.py
    ```

3. The script will process the video, detect parking slots, and display a window with the video feed where empty slots are highlighted in green and occupied slots in red. The number of available slots will be displayed on the video.

## Explanation of Files
- **model.py**: The main script that processes the video feed, detects parking slots, and determines their occupancy status.
- **utils.py**: Contains utility functions such as loading the machine learning model, resizing images, and extracting parking slot bounding boxes from the mask.
- **input_data/**: Directory containing the video file and the mask file.
  - **mask_1920_1080.png**: Binary mask image used to identify the parking slots.
  - **parking_1920_1080_loop.mp4**: Video file of the parking lot.
- **parking_classifier/**: Directory containing the pre-trained SVC model.
  - **model.p**: Pre-trained SVC model used to classify parking slots as empty or occupied.

## Acknowledgements
This project utilizes OpenCV for video processing and computer vision tasks, scikit-learn for the machine learning model, and scikit-image for image processing. The project structure and approach are inspired by common practices in the field of computer vision.

---

Feel free to contribute to this project by submitting issues or pull requests. If you have any questions or need further assistance, please don't hesitate to contact me at [Email](mailto:aletinikhilreddy759@gmail.com).

Happy coding!
