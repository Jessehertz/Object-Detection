from ultralytics import YOLO

def train_yolo_model(config_path, model_path='yolov8n.pt', epochs=50, imgsz=640):

    model = YOLO(model_path)


    results = model.train(data=config_path, epochs=epochs, imgsz=imgsz)

    return results

if __name__ == "__main__":

    config_path = 'config.yaml'


    results = train_yolo_model(config_path)
