import json

# 1️⃣ 读取 GT 和预测
gt_file = "/home/littlecoke/Desktop/COCO_distraction/annotations/distraction_val.json"
pred_file = "runs/detect/val_yolo11n/predictions.json"
pred_fixed_file = "runs/val/predictions_cocoeval.json"


gt = json.load(open(gt_file))
pred = json.load(open(pred_file))

# 2️⃣ 建立 file_name -> image_id 映射
fname2id = {img["file_name"]: img["id"] for img in gt["images"]}

# 3️⃣ 修正预测的 image_id
for p in pred:
    fname = p["file_name"]
    p["image_id"] = fname2id[fname]
    p.pop("file_name", None)  # COCOeval 不需要 file_name

# 4️⃣ 保存新的 JSON
with open(pred_fixed_file, "w") as f:
    json.dump(pred, f)

print(f"Fixed predictions saved to {pred_fixed_file}")
