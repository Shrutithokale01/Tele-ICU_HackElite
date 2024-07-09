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
2.  Download [best.py](https://github.com/Shrutithokale01/Tele-ICU_HackElite/blob/main/best.pt) file provided.
3. Download high resolution mp4 ICU room videos provided.
4. Download Audio file [Audiofile.mp3](https://github.com/Shrutithokale01/Tele-ICU_HackElite/blob/main/Audiofile.mp3) provided.
5. Create an Output folder to save detected frames.

## In Pycharm or Jupyter

1. Clone the code [3 model integrated + GUI.py](https://github.com/Shrutithokale01/Tele-ICU_HackElite/blob/main/3%20model%20integrated%20%2B%20GUI.py).
2. Importing libraries.
3. Adjust the path of [best.py](https://github.com/Shrutithokale01/Tele-ICU_HackElite/blob/main/best.pt) file provided.
4. Adjust the path of 'Output folder'.
5. Adjust the path of sound file [Audiofile.mp3](https://github.com/Shrutithokale01/Tele-ICU_HackElite/blob/main/Audiofile.mp3) provided.

## Working of GUI

1. User will run the code
2. GUI pops up
   
![GUI](https://github.com/Shrutithokale01/Tele-ICU_HackElite/blob/main/GUI.png)

3. Select the video path using the upload button.
4. Select the Output folder to save the detected frames using the process button.
5. The detection will start by running the video and the detected frames will be saved in the output folder.

## Individual  working of 3 models


| video link      | Code for the 3 models            | Description                                       |
|:---------------:|:--------------------------------:|:------------------------------------------------:|
| link            | [object detection.py](https://github.com/Shrutithokale01/Tele-ICU_HackElite/blob/main/object%20detection.py).  | This code will only run the object detection model. |
| link            | [keypointdetection.py](https://github.com/Shrutithokale01/Tele-ICU_HackElite/blob/main/keypointdetection%20.py)   | This code will only run the keypointdetection model. |
| link            | [motion yes or no.py](https://github.com/Shrutithokale01/Tele-ICU_HackElite/blob/main/motion%20yes%20or%20no%20.py)   | This code will only run the Motion detection yes or no model. |


## Project Videos

#### Video 1
See how the motion detection model identifies patient activity in the ICU:

[![Video 1](https://github.com/Shrutithokale01/Tele-ICU_HackElite/blob/main/Thumbnail/yes%20no%20thumbnail.png)](https://www.youtube.com/watch?v=xhHx4TTqLhY)

#### Video 2
Watch the video demonstration of our object detection model in the ICU room:

[![Video 2](https://github.com/Shrutithokale01/Tele-ICU_HackElite/blob/main/Thumbnail/Object%20thumbnail.png)](https://www.youtube.com/watch?v=iSS_8ZF7QJM)

#### Video 3
Explore the keypoint detection and alerting model in action:

[![Video 3](https://github.com/Shrutithokale01/Tele-ICU_HackElite/blob/main/Thumbnail/Keypoint%20thumbnail.png)](https://www.youtube.com/watch?v=HB3K9wKym50)

### Powerpoint presentation link

You can view the project presentation [intel.pptx](https://github.com/Shrutithokale01/Tele-ICU_HackElite/blob/main/intel.pptx).


## CONCLUSION

The Innovative Monitoring System for TeleICU Patients project demonstrates the power of integrating video processing and deep learning techniques to enhance patient care in ICU settings. By developing models for object detection, motion detection, and keypoint detection, we have created a comprehensive system capable of real-time monitoring and analysis. The integration of these models, along with a user-friendly GUI, ensures seamless operation and valuable insights for healthcare providers. This project not only improves patient safety and care but also optimizes the efficiency of medical staff, paving the way for future advancements in teleICU technology.


### Members

1. Alfiya Shaikh
2. Rupsha Sarkar
3. Shruti Thokale
4. Vashi Gathole
