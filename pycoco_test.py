# -*- coding:utf-8 -*-
# @Time: 2021/11/2311:16
# @Author: StevenX
# @Description:
import argparse
import json
import os
import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
import json

def test_coco():

    pred_json = str("/home/fofo/A/xsq/yolov5_rotation_anchore_free_decoupled_centernet/runs/val/exp4/YOLOv5_DOTAv1.5_OBB_predictions.json")  # annotations json
    # pred_json = str("/home/fofo/A/xsq/datasets/dota-v15/val2021/annotation/dota_val_v15.json")  # annotations json
    anno_json = str("/home/fofo/A/xsq/datasets/dota-v15/non-normal-dota/val_dota/annotation/rotate_val.json")

    # with open(anno_json, 'r') as f:
    #     anno_data = json.load(f)

    # TODO:保存为JSON格式
    with open(pred_json, 'r') as f:
        pred_data = json.load(f)
    # with open("/home/fofo/A/xsq/yolov5_rotation_anchore_free_decoupled_centernet/detect_anntations/yolov5_pred_json.json", 'w') as f:
    #     json.dump(pred_data,f)
    #
    # for anno in anno_data['annotations']:
    #     print(anno['image_id'])
    #
    #


    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        anno = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        eval = COCOeval(anno, pred, 'bbox')

        # if is_coco:
        #     eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    except Exception as e:
        print(f'pycocotools unable to run: {e}')

if __name__=='__main__':
    test_coco()