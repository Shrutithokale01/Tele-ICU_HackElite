#only detects patient and detects its motion
#yes or no
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def calculate_euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Load the trained YOLO model for inference
model = YOLO(r"C:\Users\HP\runs\detect\train7\weights\best.pt")  # Adjust the path if necessary

def draw_boxes(frame, results):
    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        clss = result.boxes.cls

        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            if label == 'patient':  # Only draw boxes for the patient
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    motion_threshold = 5000  # Adjust threshold for motion detection

    # Initialize video capture from a video file
    path_of_video = r"D:\intel project\Advanced Critical Care Nursing_ General Assessment (1).mp4" # Adjust the path to your video file
    capture = cv2.VideoCapture(path_of_video)
    if not capture.isOpened():
        print("Error: Could not open video.")
        return

    # Get the video writer initialized to save the output video
    outputvideo_path = r"D:\intel project\output\video.mp4"  # Adjust the path to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(outputvideo_path, fourcc, 30.0, (int(capture.get(3)), int(capture.get(4))))

    # Initialize previous frame for motion detection
    ret, previous_frame = capture.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    previous_frame = cv2.GaussianBlur(previous_frame, (21, 21), 0)

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

            # Convert current frame to grayscale for motion detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

            # Compute the absolute difference between the current frame and the previous frame
            frame_delta = cv2.absdiff(previous_frame, gray_frame)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Check if motion is detected
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
                    if cv2.contourArea(contour) > motion_threshold:
                        x, y, w, h = cv2.boundingRect(contour)
                        if x1 < x < x2 and y1 < y < y2:
                            motion_detected = True
                            break

            # Update the previous frame for the next iteration
            previous_frame = gray_frame.copy()

            # Draw bounding boxes around detected objects
            draw_boxes(image, yolo_results)

            # Draw face landmarks
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                )

            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                )

            # Draw right hand landmarks
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                )

            # Draw left hand landmarks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                )

            # Display motion detection alert
            if motion_detected:
                cv2.putText(image, 'Alert: Motion Detected!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, 'Alert: No Motion Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save the frame to the output video
            video_out.write(image)

            # Display the frame with detections
            cv2.imshow('Video Feed with MediaPipe and YOLO', image)
            if cv2.waitKey(10) & 0xFF == 27:
                break

    capture.release()
    video_out.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()