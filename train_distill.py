import warnings

warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller

if __name__ == '__main__':
    param_dict = {
        # student
        'model': 'ultralytics/cfg/models/11/yolo11n-C3k2_UIB(backbone)+hyper_head.yaml',
        'data': 'data.yaml',
        'imgsz': 640,
        'epochs': 300,
        'batch': 8,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '1',
        'close_mosaic': 20,
        'amp': False,  # 如果蒸馏损失为nan，请把amp设置为False
        'project': 'runs/distill',
        'name': 'YOLO11n_C3k2_UIB_hyper_head(YOLO11m_distill)_pretrained_yolo11n.pt',

        # teacher
        'teacher_weights': 'runs/detect/train_yolo11m_pretrained/weights/best.pt',
        'teacher_cfg': 'ultralytics/cfg/models/11/yolo11m.yaml',
        'kd_loss_type': 'all',
        'kd_loss_decay': 'constant',

        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 0.8,

        'teacher_kd_layers': '16,19,22',
        'student_kd_layers': '26,29,32',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 0.5
    }

    model = DetectionDistiller(overrides=param_dict)
    model.distill("runs/detect/train_yolo11n_pretrained/weights/best.pt")
