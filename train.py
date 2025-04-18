from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n-seg.pt') 

    # Train the model
    results = model.train(
        data='yolo_dataset.yml',
        epochs=100,
        imgsz=640,
        batch=8,
        name='yoloseg'
    )


