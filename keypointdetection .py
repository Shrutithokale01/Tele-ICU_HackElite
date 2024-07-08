# final keypoint detection
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def calculate_euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) * 2 + (point1[1] - point2[1]) * 2)


# load the yolov8 model
model = YOLO(r"C:\Users\HP\runs\detect\train7\weights\best.pt")


# (r"C:\Users\DELL\PycharmProjects\pythonProject\.venv\Lib\runs\detect\train4\weights\best.pt")

# to draw bounding box around detected object in a frame
def draw_boxes(frame, results):
    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        clss = result.boxes.cls

        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# function for patient box

def within_patient_box(x, y, patient_boxes):
    for (x1, y1, x2, y2) in patient_boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def main():
    mouth_open_threshold_value = 5
    eye_open_threshold_value = 5
    head_movement_threshold_value = 5
    hand_movement_threshold_value = 5

    # vido path
    path_of_video = r"D:\intel project\Advanced Critical Care Nursing_ General Assessment (1).mp4"  # Adjust the path to your video file
    capture = cv2.VideoCapture(path_of_video)
    if not capture.isOpened():
        print("Error: Could not open video.")
        return

    # path to save the output video
    outputvideo_path = r"D:\intel project\output\video.mp4"  # Adjust the path to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(outputvideo_path, fourcc, 30.0, (int(capture.get(3)), int(capture.get(4))))

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

            # Initialize variables for different movements
            mouthopen_detected = False
            eyesopen_detected = False
            headmoving_detected = False
            righthand_moving_detected = False
            lefthand_moving_detected = False

            # bounding boxes of patients
            patient_boxes = [
                (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                for box, cls in zip(yolo_results[0].boxes.xyxy, yolo_results[0].boxes.cls)
                if model.names[int(cls)] == "patient"
                # Replace "patient" with the actual label name of the patient class
            ]

            if results.face_landmarks:
                # mouth open function
                top_lip = results.face_landmarks.landmark[13]
                bottom_lip = results.face_landmarks.landmark[14]
                mouth_opendistance = calculate_euclidean_distance(
                    (top_lip.x * width, top_lip.y * height),
                    (bottom_lip.x * width, bottom_lip.y * height)
                )
                if mouth_opendistance > mouth_open_threshold_value:
                    if within_patient_box(top_lip.x * width, top_lip.y * height, patient_boxes):
                        mouthopen_detected = True

                #  eyes open function
                lefteye_top = results.face_landmarks.landmark[159]
                lefteye_bottom = results.face_landmarks.landmark[145]
                lefteye_opendistance = calculate_euclidean_distance(
                    (lefteye_top.x * width, lefteye_top.y * height),
                    (lefteye_bottom.x * width, lefteye_bottom.y * height)
                )
                righteye_top = results.face_landmarks.landmark[386]
                righteye_bottom = results.face_landmarks.landmark[374]
                righteye_opendistance = calculate_euclidean_distance(
                    (righteye_top.x * width, righteye_top.y * height),
                    (righteye_bottom.x * width, righteye_bottom.y * height)
                )
                if lefteye_opendistance > eye_open_threshold_value and righteye_opendistance > eye_open_threshold_value:
                    if within_patient_box(lefteye_top.x * width, lefteye_top.y * height, patient_boxes):
                        eyesopen_detected = True

                #  head movement function
                nose_tip = results.face_landmarks.landmark[1]
                current_headposition = (nose_tip.x * width, nose_tip.y * height)
                if previous_headposition is not None:
                    head_movementdistance = calculate_euclidean_distance(previous_headposition, current_headposition)
                    if head_movementdistance > head_movement_threshold_value:
                        if within_patient_box(nose_tip.x * width, nose_tip.y * height, patient_boxes):
                            headmoving_detected = True
                previous_headposition = current_headposition

            def check_handmovement(current_hand_landmarks, previous_hand_landmarks, hand_name):
                nonlocal righthand_moving_detected, lefthand_moving_detected
                if previous_hand_landmarks:
                    hand_movementdistance = calculate_euclidean_distance(
                        (current_hand_landmarks[0].x * width, current_hand_landmarks[0].y * height),
                        (previous_hand_landmarks[0].x * width, previous_hand_landmarks[0].y * height)
                    )
                    if hand_movementdistance > hand_movement_threshold_value:
                        if hand_name == "Right" and within_patient_box(current_hand_landmarks[0].x * width,
                                                                       current_hand_landmarks[0].y * height,
                                                                       patient_boxes):
                            righthand_moving_detected = True
                        elif hand_name == "Left" and within_patient_box(current_hand_landmarks[0].x * width,
                                                                        current_hand_landmarks[0].y * height,
                                                                        patient_boxes):
                            lefthand_moving_detected = True

            if results.right_hand_landmarks:
                check_handmovement(results.right_hand_landmarks.landmark, previous_righthand_position, "Right")
                previous_righthand_position = results.right_hand_landmarks.landmark

            if results.left_hand_landmarks:
                check_handmovement(results.left_hand_landmarks.landmark, previous_lefthand_position, "Left")
                previous_lefthand_position = results.left_hand_landmarks.landmark

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

            # Display detected movements
            if mouthopen_detected:
                cv2.putText(image, 'Alert: Mouth is open!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if eyesopen_detected:
                cv2.putText(image, 'Alert: Eyes are open!', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if headmoving_detected:
                cv2.putText(image, 'Alert: Head is moving!', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if righthand_moving_detected:
                cv2.putText(image, 'Alert: Right hand is moving!', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)
            if lefthand_moving_detected:
                cv2.putText(image, 'Alert: Left hand is moving!', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)

            # save the frames only
            video_out.write(image)

            # frame with detections
            cv2.imshow('Video Feed with MediaPipe and YOLO', image)
            if cv2.waitKey(10) & 0xFF == 27:
                break

    capture.release()
    video_out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()