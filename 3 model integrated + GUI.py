#voice integration gui  
#importing necessary libraries
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pygame

# Initializing MediaPipe Holistic Model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initializing Pygame Mixer
pygame.mixer.init()

# Loading the sound file for motion detecetd
soundfile_path = r"C:\Users\Rupsha Sarkar\Downloads\motion detected 1.mp3"
pygame.mixer.music.load(soundfile_path)

#To calculate the euclidean distance between a 2d space(between 2 points)
def calculate_euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) * 2 + (point1[1] - point2[1]) * 2)

# Load the trained YOLO model for inference and give the path to best.pt file
model = YOLO(r"C:\Users\Rupsha Sarkar\Downloads\best.pt")

# To draw bounding boxes around detected objects in a frame
def draw_boxes(frame, results):
    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        clss = result.boxes.cls

        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            color = (0, 255, 0)  # Green color for patient
            if label == 'doctor':
                color = (255, 0, 0)  # Blue color for doctor
            elif label == 'nurse':
                color = (0, 0, 255)  # Red color for nurse

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#Function for patient box
def within_patient_box(x, y, patient_boxes):
    for (x1, y1, x2, y2) in patient_boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

#Function to process the video
def process_video(video_path, output_folder):
    motion_threshold_value = 5000  # Add threshold value for motion detection
    mouth_open_threshold_value = 5
    eye_open_threshold_value = 5
    head_movement_threshold_value= 5
    hand_movement_threshold_value = 5

    capture = cv2.VideoCapture(video_path)   #video capturing
    if not capture.isOpened():
        messagebox.showerror("Error", "Could not open the video.") #if video cannot open,gives error
        return

    if not os.path.exists(output_folder): #If output folder dont exists,creates a new output folder
        os.makedirs(output_folder)

    frame_count = 0  #Initial frame count is 0

    ret, previous_frame = capture.read()
    if not ret:
        messagebox.showerror("Error", "Could not read the first frame.")
        return
    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    previous_frame = cv2.GaussianBlur(previous_frame, (21, 21), 0)

    previous_headposition = None
    previous_lefthand_position = None
    previous_righthand_position = None

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yolo_results = model(image)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            height, width, _ = image.shape

            #converts the input frame from color image to gray scale image
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

            frame_delta = cv2.absdiff(previous_frame, gray_frame)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            patient_detected = False
            patient_box = None

            for result in yolo_results:
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    if model.names[int(cls)] == 'patient':
                        x1, y1, x2, y2 = map(int, box)
                        patient_detected = True
                        patient_box = (x1, y1, x2, y2)
                        break

            if patient_detected and patient_box:
                x1, y1, x2, y2 = patient_box
                for contour in contours:
                    if cv2.contourArea(contour) > motion_threshold_value:
                        x, y, w, h = cv2.boundingRect(contour)
                        if x1 < x < x2 and y1 < y < y2:
                            motion_detected = True
                            break

            previous_frame = gray_frame.copy()

            if motion_detected:
                pygame.mixer.music.play()  # Here when motion of patient is detecetd it will give an alert by playing sound

            #Initial detection  is set to false
            mouthopen_detected = False
            eyesopen_detected = False
            headmoving_detected = False
            righthand_moving_detected = False
            lefthand_moving_detected = False

            patient_boxes = [
                (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                for box, cls in zip(yolo_results[0].boxes.xyxy, yolo_results[0].boxes.cls)
                if model.names[int(cls)] == "patient"
            ]

            #for face
            if results.face_landmarks:
                top_lip = results.face_landmarks.landmark[13]
                bottom_lip = results.face_landmarks.landmark[14]
                mouth_opendistance = calculate_euclidean_distance(
                    (top_lip.x * width, top_lip.y * height),
                    (bottom_lip.x * width, bottom_lip.y * height)
                )

            #for mouth
                if mouth_opendistance > mouth_open_threshold_value:
                    if within_patient_box(top_lip.x * width, top_lip.y * height, patient_boxes):
                        mouth_open_detected = True

            #left eye
                lefteye_top = results.face_landmarks.landmark[159]
                lefteye_bottom = results.face_landmarks.landmark[145]
                lefteye_opendistance = calculate_euclidean_distance(
                    (lefteye_top.x * width, lefteye_top.y * height),
                    (lefteye_bottom.x * width, lefteye_bottom.y * height)
                )
            #right eye
                righteye_top = results.face_landmarks.landmark[386]
                righteye_bottom = results.face_landmarks.landmark[374]
                righteye_opendistance = calculate_euclidean_distance(
                    (righteye_top.x * width, righteye_top.y * height),
                    (righteye_bottom.x * width, righteye_bottom.y * height)
                )
                if lefteye_opendistance > eye_open_threshold_value and righteye_opendistance > eye_open_threshold_value:
                    if within_patient_box(lefteye_top.x * width, lefteye_top.y * height, patient_boxes):
                        eyes_open_detected = True

            #for nose
                nose_tip = results.face_landmarks.landmark[1]
                current_headposition = (nose_tip.x * width, nose_tip.y * height)
                if previous_headposition is not None:
                    head_movementdistance = calculate_euclidean_distance(previous_headposition, current_headposition)
                    if head_movementdistance > head_movement_threshold_value:
                        if within_patient_box(nose_tip.x * width, nose_tip.y * height, patient_boxes):
                            head_moving_detected = True
                previous_headposition = current_headposition


            #Function to check hand movements
            def check_handmovement(current_hand_landmarks, previous_hand_landmarks, hand_name):
                nonlocal righthand_moving_detected, lefthand_moving_detected
                if previous_hand_landmarks:
                    hand_movementdistance = calculate_euclidean_distance(
                        (current_hand_landmarks[0].x * width, current_hand_landmarks[0].y * height),
                        (previous_hand_landmarks[0].x * width, previous_hand_landmarks[0].y * height)
                    )
                    if hand_movementdistance > hand_movement_threshold_value:
                        if hand_name == "Right" and within_patient_box(current_hand_landmarks[0].x * width, current_hand_landmarks[0].y * height, patient_boxes):
                            righthand_moving_detected = True
                        elif hand_name == "Left" and within_patient_box(current_hand_landmarks[0].x * width, current_hand_landmarks[0].y * height, patient_boxes):
                            lefthand_moving_detected = True

            if results.right_hand_landmarks:
                check_handmovement(results.right_hand_landmarks.landmark, previous_righthand_position, "Right")
                previous_righthand_position = results.right_hand_landmarks.landmark

            if results.left_hand_landmarks:
                check_handmovement(results.left_hand_landmarks.landmark, previous_lefthand_position, "Left")
                previous_lefthand_position = results.left_hand_landmarks.landmark


            draw_boxes(image, yolo_results)

            #Gives alerts as the motion is detected
            if motion_detected:
                cv2.putText(image, 'Motion Detected', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if mouthopen_detected:
                cv2.putText(image, 'Alert: Mouth Open Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if eyesopen_detected:
                cv2.putText(image, 'Alert: Eyes Open Detected', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if headmoving_detected:
                cv2.putText(image, 'Alert: Head Moving Detected', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if righthand_moving_detected:
                cv2.putText(image, 'Alert: Right Hand Moving Detected', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if lefthand_moving_detected:
                cv2.putText(image, 'Alert: Left Hand Moving Detected', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if results.face_landmarks:
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


            frame_count += 1
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, image)
            cv2.imshow('Frame', image)
            
   

            if cv2.waitKey(1) & 0xFF == 27:
                break

    capture.release()
    cv2.destroyAllWindows()

#Function to upload the video using upload button
def upload_video_button():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", ".mp4;.avi;.mov;.mkv")])
    if file_path:
        video_path.set(file_path)

#Function to process the video using upload button
def process_video_button():
    path_of_video = video_path.get()
    if not path_of_video:
        messagebox.showerror("Error", "Please upload a video first.")
        return

    output_folder = filedialog.askdirectory()
    if output_folder:
        process_video(path_of_video, output_folder)
        messagebox.showinfo("Success", f"Frames processed and saved to: {output_folder}")

gui = tk.Tk()
gui.title("Video Processing Application")   #title
gui.geometry("400x300")
gui.configure(bg="#f0f0f0")

video_path = tk.StringVar()

heading = tk.Label(gui, text="VIDEO PROCESSING GUI", font=("Anton SC", 25, "bold"), bg="#f0f0f0", pady=22)
heading.pack()

#upload button
uploadvideo_button = tk.Button(gui, text="Upload Video", command=upload_video_button, width=22, height=4, bg="#9A32CD",
                              fg="#FFFFFF", font=("Helvetica", 16),highlightbackground="black",
                   highlightcolor="green",
                   highlightthickness=2,
                   justify="center",
                   overrelief="raised")
uploadvideo_button.pack(pady=12)

#process button
processvideo_button = tk.Button(gui, text="Process Video", command=process_video_button, width=22, height=4, bg="#1874CD",
                               fg="#FFFFFF", font=("Helvetica", 16),cursor="hand2",
                   disabledforeground="gray",
                   highlightbackground="black",
                   highlightcolor="green",
                   highlightthickness=2,
                   justify="center",
                   overrelief="raised",)
processvideo_button.pack(pady=12)

gui.mainloop()
