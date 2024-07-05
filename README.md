# Intel Unnati Industrial Training Program
## Problem Statement - Innovative Monitoring System for TeleICU Patients using Video Processing and Deep learning
### INTRODUCTION - 

- TeleICU is concept for monitoring ICU patients from remote locations to reduce the burden of on-site intensivist.
- The proposed solution should work to reduce the burden of remote health care professional so, one remote health care professional can monitor 5 or more patients at single time.

### OBJECTIVES – 

The primary objective of this project is to develop a robust and accurate monitoring system for TeleICU settings, focusing on two key aspects:

- Object Detection: Train a deep learning model to accurately identify various individuals present in the ICU room, including Nurses, Doctors and Patients. This will help in ensuring proper care and monitoring of patient interactions.
- Patient Motion Recognition: Develop a deep learning model to recognize and categorize different activities of the patient when they are alone. This includes monitoring for signs of distress, abnormal movements, and other critical activities that require immediate attention.

### AIM –

- Enhance patient safety through continuous and automated monitoring.
- Assist healthcare providers in timely intervention and decision-making.
- Improve the overall efficiency and effectiveness of ICU operations.


## Tools and Technologies:


- Object Detection Models: YOLOv8
- Development environment (IDE): Pycharm, Jupyter
- Motion Recognition/ tracking: mediapipe
- Data Annotation Tools: roboflow
- Video Processing Libraries: OpenCV
- GUI: Tkinter
- Other: Pygame, numpy, os

## Unique Idea Brief -

### OUR APPROACH:

We have taken 2-3 videos of the ICU Room .
To work on the video and to detect the functionality of the patient , we first created the dataset using ROBOFLOW - which mainly include set of data’s for (Doctors, Nurse and  Patient)
Now we have used three different deep learning and video processing techniques and models to build our project

	     Object Detection Model

	     Motion detecttion(yes or no) model

	     Keypoint detection + alerting model

Then we have integrated the three models in order to simplify user experience.
Then we have also included the GUI where the user need to select the saved video and then process that video to get the integrated model working on the video.

## PREREQUISITES

(To run '3 model integrated + GUI.py') follow the steps given below-

1. Install necessary libraries.
```bash
pip install ultralytics
pip install jupyterlab
pip install mediapipe
pip install opencv-python
pip install pygame
pip install numpy
```
2.  Download best.pt file provided.
3. Download high resolution mp4 ICU room videos provided.
4. Download Audio file 'Audio.mp3' provided.
5. Create an Output folder to save detected frames.

## In Pycharm

1. Clone the code '3 model integrated + GUI.py'.
2. Importing libraries.
3. Adjust the path of 'best.pt' file provided.
4. Adjust the path of 'Output folder'.
5. Adjust the path of sound file 'Audio.mp3' provided.

[3 model integrated + GUI.py]()

