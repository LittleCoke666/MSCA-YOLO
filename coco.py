from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt_file = "/home/littlecoke/Desktop/COCO_distraction/annotations/distraction_val.json"
pred_file = "runs/val/predictions_cocoeval.json"

coco_gt = COCO(gt_file)
coco_dt = coco_gt.loadRes(pred_file)

coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()


