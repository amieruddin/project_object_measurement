from ultralytics import YOLO

import cv2
import torch

video_path  = "./source/object_on_table.mp4"
model_path  = "./models/yolo11s-seg.pt"
target_class= [63,67]
def load_model():
    model = YOLO(model_path)
    return model


if __name__ == "__main__":

    model=load_model()

    # Open video file
    video_path = video_path 
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Perform inference
        results = model(frame, classes=target_class)

        # Display the results with bounding boxes
        annotated_frame = results[0].plot()
        resize_frame = cv2.resize(annotated_frame,(480,720))

        cv2.imshow("YOLO Inference", resize_frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release video and close windows
    cap.release()
    cv2.destroyAllWindows()



