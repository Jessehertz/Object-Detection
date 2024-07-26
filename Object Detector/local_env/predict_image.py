import os
import cv2
from ultralytics import YOLO

def detect_objects_in_image(image_path, output_path, model_path, threshold=0.3):

    model = YOLO(model_path)


    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image file: {image_path}")


    results = model(image)

    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box

        if score > threshold:

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imwrite(output_path, image)
    print(f"Image processing completed. Output saved at {output_path}")

if __name__ == "__main__":

    IMAGES_DIR = r'C:\Users\adeda\Downloads\train-yolov8-custom-dataset-step-by-step-guide-master\train-yolov8-custom-dataset-step-by-step-guide-master\local_env\images'
    image_path = os.path.join(IMAGES_DIR, 'car.jpg')
    output_path = os.path.splitext(image_path)[0] + '_out.jpg'
    model_path = r'runs\detect\train\weights\best.pt'


    detect_objects_in_image(image_path, output_path, model_path, threshold=0.3)
