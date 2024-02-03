# Drone Suiveur

## Description
This is an end of year school project that aims to train a drone to autonomously follow a human using the SDK Tello drone and the AS-ONE library for YOLO object detection. By leveraging computer vision techniques, the drone will be able to detect and track a human in real-time, allowing for dynamic and interactive aerial movement.

## Features
- Human Detection: The project utilizes the AS-ONE library, which is based on YOLO (You Only Look Once), to detect humans in the drone's video feed. This enables the drone to identify and track a human with accuracy and efficiency.
- Autonomous Flight: Once a human is detected, the drone will autonomously adjust its flight path to follow the human's movements. It employs control algorithms to maintain a consistent distance and angle, ensuring smooth and stable tracking.
- Real-Time Feedback: The project provides real-time feedback by displaying the drone's video feed with bounding boxes around detected humans. This visual representation allows users to verify the accuracy of the detection and tracking process.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
It is imperative to install the requirements for AS-One (follow the guide at https://github.com/augmentedstartups/AS-One), it depends on your operating system (for windows, it is recommended to use a miniconda environment with python 3.9.x).

You must install the following packages with pip, in addition with the packages for AS-One (and yolo): djitellopy, numpy, opencv

If you want to retrain the tracking model (using yolov5), follow the guide for yolo installation at https://github.com/ultralytics/yolov5.

## Usage
To begin tracking with the drone: run the script drone-suiveur/AS-One/main.py

You can additionnally train the classifier CNN (and even change it) with the train1.py scripts of the python directory, and a custom dataset (with the persons you want to follow as positive class 1, and the others as negative class 0).

You can retrain yolov5 model within the yolov5 directory (follow the guide for training yolo at https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/), then the newly train weights can be used by the AS-One library (by changing the main.py file in AS-One directory).

## Roadmap
Left to be implemented :
- smoother movement
- optimizing detection and sending commands for faster and smoother drone movement
- implementing algorithms that allow the drone to follow its target even after disappearing (taking sharp turn into a hallway)
- adding a user friendly interface on top of the video feedback
- adding possibilities for user to train the image detction software to only detect themn thus ensuring that the drone will only follow them

## Problems you will encounter

- Tello is very slow when it moves (only 1 move every 2 to 5 seconds, depending on the distance traveled, and the stabilizing needed after the move), it also has some issues with its inertial measurement unit you may need to recalibrate.
- Tello's camera stream has issues to be decoded with FFMPEG. You will need to unburden the thread decoding the camera stream (in AS-One/asone/asone.py, in the __track_stream__ method), or compile Opencv with a better decoder that FFMPEG (but you may enconter issues with yolo and then AS-One because of the version you will have).
- You can improve the scope of movements by using image based visual servoing (https://robotacademy.net.au/lesson/image-based-visual-servoing/), which a beggining of implementation is in AS-One/movement.py. It will need heavy testing and improments in the performances of the tracking and in the speed at which the drone can perform actions.

## Authors and acknowledgment
Experts that guided us through the project : 
-Sumanta CHAUDHURI
-Enzo TARTAGLIONE

Students that worked on the project :
-Roan RUBIALES
-Lucas CHEVRIER
-Meriem NAJI
-Mohamed AHMED BOUHA

Special thanks to the creators and contributors of the AS-ONE library and the Tello SDK :
    AS-ONE: https://github.com/augmentedstartups/AS-One
    Tello SDK: https://github.com/dji-sdk/Tello-Python

## Project status
Project development is completely stopped right now.
