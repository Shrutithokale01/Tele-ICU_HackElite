#voice integration gui  01/07/2024

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
soundfile_path = r"C:\Users\DELL\Downloads\motion detected 1.mp3"
pygame.mixer.music.load(soundfile_path)

#To calculate the euclidean distance between a 2d space(between 2 points)
def calculate_euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) * 2 + (point1[1] - point2[1]) * 2)

# Load the trained YOLO model for inference and give the path to best.pt file
model = YOLO(r"C:\Users\DELL\Downloads\best (1).pt")

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

def within_patient_box(x, y, patient_boxes):
    for (x1, y1, x2, y2) in patient_boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

def process_video(video_path, output_folder):
    motion_threshold_value = 5000  # Add threshold value for motion detection
    mouth_open_threshold_value = 5
    eye_open_threshold_value = 5
    head_movement_threshold_value= 5
    hand_movement_threshold_value = 5

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0

    ret, previous_frame = capture.read()
    if not ret:
        messagebox.showerror("Error", "Could not read the first frame.")
        return
    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    previous_frame = cv2.GaussianBlur(previous_frame, (21, 21), 0)

    previous_head_position = None
    previous_left_hand_position = None
    previous_right_hand_position = None

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
                pygame.mixer.music.play()  # Play the sound alert

            mouth_open_detected = False
            eyes_open_detected = False
            head_moving_detected = False
            right_hand_moving_detected = False
            left_hand_moving_detected = False

            patient_boxes = [
                (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                for box, cls in zip(yolo_results[0].boxes.xyxy, yolo_results[0].boxes.cls)
                if model.names[int(cls)] == "patient"
            ]

            if results.face_landmarks:
                top_lip = results.face_landmarks.landmark[13]
                bottom_lip = results.face_landmarks.landmark[14]
                mouth_open_distance = calculate_euclidean_distance(
                    (top_lip.x * width, top_lip.y * height),
                    (bottom_lip.x * width, bottom_lip.y * height)
                )
                if mouth_open_distance > mouth_open_threshold_value:
                    if within_patient_box(top_lip.x * width, top_lip.y * height, patient_boxes):
                        mouth_open_detected = True

                left_eye_top = results.face_landmarks.landmark[159]
                left_eye_bottom = results.face_landmarks.landmark[145]
                left_eye_open_distance = calculate_euclidean_distance(
                    (left_eye_top.x * width, left_eye_top.y * height),
                    (left_eye_bottom.x * width, left_eye_bottom.y * height)
                )
                right_eye_top = results.face_landmarks.landmark[386]
                right_eye_bottom = results.face_landmarks.landmark[374]
                right_eye_open_distance = calculate_euclidean_distance(
                    (right_eye_top.x * width, right_eye_top.y * height),
                    (right_eye_bottom.x * width, right_eye_bottom.y * height)
                )
                if left_eye_open_distance > eye_open_threshold_value and right_eye_open_distance > eye_open_threshold_value:
                    if within_patient_box(left_eye_top.x * width, left_eye_top.y * height, patient_boxes):
                        eyes_open_detected = True

                nose_tip = results.face_landmarks.landmark[1]
                current_head_position = (nose_tip.x * width, nose_tip.y * height)
                if previous_head_position is not None:
                    head_movement_distance = calculate_euclidean_distance(previous_head_position, current_head_position)
                    if head_movement_distance > head_movement_threshold_value:
                        if within_patient_box(nose_tip.x * width, nose_tip.y * height, patient_boxes):
                            head_moving_detected = True
                previous_head_position = current_head_position

            def check_hand_movement(current_hand_landmarks, previous_hand_landmarks, hand_name):
                nonlocal right_hand_moving_detected, left_hand_moving_detected
                if previous_hand_landmarks:
                    hand_movement_distance = calculate_euclidean_distance(
                        (current_hand_landmarks[0].x * width, current_hand_landmarks[0].y * height),
                        (previous_hand_landmarks[0].x * width, previous_hand_landmarks[0].y * height)
                    )
                    if hand_movement_distance > hand_movement_threshold_value:
                        if hand_name == "Right" and within_patient_box(current_hand_landmarks[0].x * width, current_hand_landmarks[0].y * height, patient_boxes):
                            right_hand_moving_detected = True
                        elif hand_name == "Left" and within_patient_box(current_hand_landmarks[0].x * width, current_hand_landmarks[0].y * height, patient_boxes):
                            left_hand_moving_detected = True

            if results.right_hand_landmarks:
                check_hand_movement(results.right_hand_landmarks.landmark, previous_right_hand_position, "Right")
                previous_right_hand_position = results.right_hand_landmarks.landmark

            if results.left_hand_landmarks:
                check_hand_movement(results.left_hand_landmarks.landmark, previous_left_hand_position, "Left")
                previous_left_hand_position = results.left_hand_landmarks.landmark

            draw_boxes(image, yolo_results)
            if motion_detected:
                cv2.putText(image, 'Motion Detected', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if mouth_open_detected:
                cv2.putText(image, 'Alert: Mouth Open Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if eyes_open_detected:
                cv2.putText(image, 'Alert: Eyes Open Detected', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if head_moving_detected:
                cv2.putText(image, 'Alert: Head Moving Detected', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if right_hand_moving_detected:
                cv2.putText(image, 'Alert: Right Hand Moving Detected', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if left_hand_moving_detected:
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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", ".mp4;.avi;.mov;.mkv")])
    if file_path:
        video_path.set(file_path)

def process_video_button():
    input_path = video_path.get()
    if not input_path:
        messagebox.showerror("Error", "Please upload a video first.")
        return

    output_folder = filedialog.askdirectory()
    if output_folder:
        process_video(input_path, output_folder)
        messagebox.showinfo("Success", f"Frames processed and saved to: {output_folder}")

app = tk.Tk()
app.title("Video Processing Application")
app.geometry("400x300")
app.configure(bg="#f0f0f0")

video_path = tk.StringVar()

headline = tk.Label(app, text="Video Processing GUI", font=("Helvetica", 18, "bold"), bg="#f0f0f0", pady=20)
headline.pack()

upload_button = tk.Button(app, text="Upload Video", command=upload_video, width=20, height=2, bg="#4caf50",
                              fg="white", font=("Helvetica", 12))
upload_button.pack(pady=10)

process_button = tk.Button(app, text="Process Video", command=process_video_button, width=20, height=2, bg="#2196f3",
                               fg="white", font=("Helvetica", 12))
process_button.pack(pady=10)

app.mainloop()