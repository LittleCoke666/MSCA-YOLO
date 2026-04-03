from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11n-hyper.yaml').load("yolo11n.pt")
    model.train(data='data.yaml',
                imgsz=640,
                epochs=300,
                batch=12,
                workers=8,
                device='0',
                )
