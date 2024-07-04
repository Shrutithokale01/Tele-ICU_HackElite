# object detection
# only required when run in jupyter
try:
    import ultralytics
except ImportError:
    !pip install ultralytics
    
# importing necessary libraries

from ultralytics import YOLO
import cv2
import os

# loading of the YOLOV8 model
model = YOLO(r"C:\Users\Rupsha Sarkar\Downloads\best.pt")  

# now mention the path of the video in which object detection apply
path_of_video = r"C:\Users\Rupsha Sarkar\Downloads\video\COVIDLAND A Film About Survival and Hope in the ICU.mp4"
capture = cv2.VideoCapture(path_of_video )

# checking of the video whether it is open successfully
if not capture.isOpened():
    print("Error: Could not open video.")
    exit()

# mention the output folder to save the frames
output_path_folder = r'D:\intell\detected_frames10'
if not os.path.exists(output_path_folder):
    os.makedirs(output_path_folder)


frame_id = 0
while capture.isOpened():
    ret, frame = capture.read() # reading of the frames from the video
    if not ret:
        break

    
    results = model.predict(source=frame)

    
    annotated_frame = results[0].plot()  

 # frames must save in the suitable format
    output_path = os.path.join(output_path_folder, f"frame_{frame_id:04d}.jpg")
    cv2.imwrite(output_path, annotated_frame)
    frame_id += 1

   # now adjust the code based on the laptop's/ screen size
    display_screen_res = 1280, 720  
    scale_width = display_screen_res[0] / frame.shape[1]
    scale_height = display_screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)

    
    cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLO Detection', window_width, window_height)
    cv2.imshow('YOLO Detection', annotated_frame)

   
    if cv2.waitKey(1) & 0xFF == 27:# exit the loop using ESC buttton
        break

#releasing out the video capture
capture.release()
cv2.destroyAllWindows()
