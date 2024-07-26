import os
import cv2
from ultralytics import YOLO

def detect_objects(video_path, output_path, model_path, threshold=0.3):

    model = YOLO(model_path)


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")


    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to read the first frame of the video.")
    H, W, _ = frame.shape


    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    while ret:

        results = model(frame)

        for result in results:
            boxes = result.boxes
            names = result.names

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                score = box.conf[0]
                class_id = box.cls[0]

                if score > threshold:

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, names[int(class_id)].upper(), (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)


        out.write(frame)


        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video processing completed. Output saved at {output_path}")

if __name__ == "__main__":

    VIDEOS_DIR = r'C:\Users\adeda\Downloads\train-yolov8-custom-dataset-step-by-step-guide-master\train-yolov8-custom-dataset-step-by-step-guide-master\local_env\videos'
    video_path = os.path.join(VIDEOS_DIR, 'car.mp4')
    output_path = os.path.splitext(video_path)[0] + '_out.mp4'
    model_path = r'C:\Users\adeda\Downloads\train-yolov8-custom-dataset-step-by-step-guide-master\train-yolov8-custom-dataset-step-by-step-guide-master\local_env\runs\detect\train\weights\best.pt'  # Path to the trained model


    detect_objects(video_path, output_path, model_path, threshold=0.3)
