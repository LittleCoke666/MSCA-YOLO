import os
import torch
from ultralytics import YOLO
import thop


def main():
    # ================= 配置 =================
    model_path = "/home/littlecoke/Desktop/Hyper-YOLO/ultralytics/models/yolo/detect/runs/train/weights/best.pt"
    image_dir = "/home/littlecoke/Desktop/YOLOv11/datasets"

    device = "cuda:1"
    imgsz = 640
    warmup_iters = 10
    # =======================================

    assert torch.cuda.is_available(), "CUDA 不可用"


    # 收集图片
    image_list = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ])

    assert len(image_list) > 0, "未找到测试图片"

    print(f"Found {len(image_list)} images.")
    print(f"Using device: {device}\n")

    # 加载模型
    print("Loading model...")
    model = YOLO(model_path)
    model.to(device)


    # ---------------- 计算Params和FLOPs ----------------
    print("\n" + "=" * 50)
    print("Model Information")
    print("=" * 50)


    # 使用thop计算FLOPs
    print("Calculating FLOPs and Parameters...")
    try:
        # 创建一个dummy输入
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
        # 获取模型的计算图
        # 备用方法：直接计算参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")
        print("Note: FLOPs calculation requires model's forward method")
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        # 使用简单方法计算参数
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total Parameters: {total_params / 1e6:.2f}M")

    # 方法3: 尝试从模型配置中获取信息
    try:
        if hasattr(model, 'info'):
            info_str = model.info()
            if info_str and 'parameters' in info_str.lower():
                print("\nModel info from YOLO:")
                print(info_str)
    except:
        pass

    # 显示模型结构摘要
    print("\n" + "-" * 50)
    try:
        # 尝试显示模型层数信息
        if hasattr(model.model, 'model'):
            num_layers = len(list(model.model.model.children()))
            print(f"Model layers: {num_layers}")
    except:
        pass

    print("=" * 50 + "\n")

    # ---------------- Warm-up ----------------
    print("Warming up GPU...")
    for _ in range(warmup_iters):
        _ = model.predict(
            source=image_list[0],
            imgsz=imgsz,
            device=device,
            verbose=False
        )

    torch.cuda.synchronize()
    print("Warm-up done.\n")

    # ---------------- Benchmark ----------------
    infer_times = []
    nms_times = []
    load_times = []

    print("Per-image timing (ms):")
    print("-" * 70)
    print(f"{'Image':30s} | {'Load':>8s} | {'Infer':>8s} | {'NMS':>8s}")
    print("-" * 70)

    for img_path in image_list:

        results = model.predict(
            source=img_path,
            imgsz=imgsz,
            device=device,
            conf=0.25,
            iou=0.45,
            save=False,
            name="",
            exist_ok=True,
            verbose=False,
        )

        r = results[0]
        speed = r.speed  # ms

        load_t = speed.get("preprocess", 0.0)
        infer_t = speed.get("inference", 0.0)
        nms_t = speed.get("postprocess", 0.0)

        load_times.append(load_t)
        infer_times.append(infer_t)
        nms_times.append(nms_t)

        print(f"{os.path.basename(img_path):30s} | "
              f"{load_t:8.2f} | {infer_t:8.2f} | {nms_t:8.2f}")

    # ---------------- Statistics ----------------
    avg_load = sum(load_times) / len(load_times)
    avg_infer = sum(infer_times) / len(infer_times)
    avg_nms = sum(nms_times) / len(nms_times)
    avg_model = avg_infer + avg_nms
    fps = 1000.0 / avg_model

    print("\n" + "=" * 50)
    print("Performance Summary")
    print("=" * 50)

    # 重新显示模型参数信息
    try:
        if hasattr(model.model, 'parameters'):
            total_params = sum(p.numel() for p in model.model.parameters())
            params_m = total_params / 1e6
            print(f"Model Params: {params_m:.2f}M")
    except:
        pass

    print(f"\nInference Statistics:")
    print(f"{'-' * 40}")
    print(f"Avg load time (ms)  : {avg_load:.2f}")
    print(f"Avg inference (ms)  : {avg_infer:.2f}")
    print(f"Avg NMS (ms)        : {avg_nms:.2f}")
    print(f"Model total (ms)    : {avg_model:.2f}")
    print(f"FPS                 : {fps:.2f}")
    print("=" * 50)

    # 额外信息：计算理论最大FPS
    if avg_infer > 0:
        theoretical_fps = 1000.0 / avg_infer
        print(f"Theoretical FPS (inference only): {theoretical_fps:.2f}")

    # 显示每个阶段的时间占比
    total_avg = avg_load + avg_infer + avg_nms
    if total_avg > 0:
        print(f"\nTime Distribution:")
        print(f"{'-' * 40}")
        print(f"Load      : {avg_load:.2f}ms ({avg_load / total_avg * 100:.1f}%)")
        print(f"Inference : {avg_infer:.2f}ms ({avg_infer / total_avg * 100:.1f}%)")
        print(f"NMS       : {avg_nms:.2f}ms ({avg_nms / total_avg * 100:.1f}%)")
        print(f"Total     : {total_avg:.2f}ms (100%)")



if __name__ == "__main__":
    # 确保thop已安装，如果没有则尝试安装
    try:
        import thop
    except ImportError:
        print("Installing thop for FLOPs calculation...")
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "thop"])
        import thop

    main()