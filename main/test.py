#!/usr/bin/python3
# coding=utf-8
import os
import sys

sys.path.insert(0, '../')
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
import torch
import torch.nn.functional as F
import argparse
import dataset
from torch.utils.data import DataLoader
from model.get_model import get_model
import datetime
import time
from tqdm import tqdm


class Test(object):
    def __init__(self, Dataset, Path, model, checkpoint, task):
        self.task = task
        self.model = model
        self.cfg = Dataset.Config(datapath=Path, snapshot=checkpoint, mode='test')
        self.data = Dataset.Data(self.cfg, model)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        self.net = get_model(self.cfg, model)
        self.net.train(False)
        self.net.cuda()

        self.mae_list = []
        self.precision_list = []
        self.recall_list = []

    def cal_mae(self, pred, gt):
        return np.mean(np.abs(pred - gt))

    def cal_precision_recall(self, pred, gt):
        pred = (pred * 255).astype(np.uint8)
        gt = (gt > 0.5).astype(np.float32)
        precision = np.zeros(256)
        recall = np.zeros(256)
        for threshold in range(256):
            pred_binary = (pred > threshold).astype(np.float32)
            tp = np.sum(pred_binary * gt)
            fp = np.sum(pred_binary * (1 - gt))
            fn = np.sum((1 - pred_binary) * gt)
            precision[threshold] = tp / (tp + fp + 1e-8)
            recall[threshold] = tp / (tp + fn + 1e-8)
        return precision, recall

    def predict_with_tta(self, image, shape):
        H, W = shape
        H, W = int(H), int(W)
        preds = []

        # 1. 原始尺寸
        out1, out2, out3, out4, _ = self.net(image, (H, W), None)
        pred = torch.sigmoid(out4[0, 0]).cpu().numpy()
        preds.append(cv2.resize(pred, (W, H)))

        # 2. 水平翻转
        flipped = torch.flip(image, dims=[3])
        out1, out2, out3, out4, _ = self.net(flipped, (H, W), None)
        pred_flip = torch.sigmoid(out4[0, 0]).cpu().numpy()
        pred_flip = np.fliplr(pred_flip)
        preds.append(cv2.resize(pred_flip, (W, H)))

        # 3. 0.75倍缩放
        scaled = F.interpolate(image, scale_factor=0.75, mode='bilinear', align_corners=False)
        out1, out2, out3, out4, _ = self.net(scaled, (H, W), None)
        pred_075 = torch.sigmoid(out4[0, 0]).cpu().numpy()
        preds.append(cv2.resize(pred_075, (W, H)))

        # 4. 1.25倍缩放
        scaled = F.interpolate(image, scale_factor=1.25, mode='bilinear', align_corners=False)
        out1, out2, out3, out4, _ = self.net(scaled, (H, W), None)
        pred_125 = torch.sigmoid(out4[0, 0]).cpu().numpy()
        preds.append(cv2.resize(pred_125, (W, H)))

        final_pred = np.mean(preds, axis=0)
        return final_pred

    def save(self):
        with torch.no_grad():
            for image, (H, W), name in tqdm(
                    self.loader,
                    desc='Processing DUTS Test Set (TTA)',
                    unit='image',
                    dynamic_ncols=True,
                    file=sys.stdout,
                    leave=True
            ):
                image = image.cuda().float()
                pred = self.predict_with_tta(image, (H, W))

                if self.task == "SOC":
                    head = 'util/evaltool/Prediction/' + self.model + '/SOC/' + self.cfg.datapath.split('/')[-1]
                else:
                    head = 'util/evaltool/Prediction/' + self.model + '/' + self.cfg.datapath.split('/')[-2]

                if not os.path.exists(head):
                    print(f"\ncreate a new folder: {head}")
                    os.makedirs(head)

                cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred * 255))

                gt_path = os.path.join(self.cfg.datapath, 'Masks', name[0] + '.png')
                if os.path.exists(gt_path):
                    gt = cv2.imread(gt_path, 0) / 255.0
                    if pred.shape != gt.shape:
                        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

                    mae = self.cal_mae(pred, gt)
                    precision, recall = self.cal_precision_recall(pred, gt)
                    self.mae_list.append(mae)
                    self.precision_list.append(precision)
                    self.recall_list.append(recall)

            print("\n" + "=" * 60)
            print("✅ 所有图片处理完成！(TTA enhanced)")
            print(f"📊 总测试图片数：{len(self.mae_list)}")
            print(f"📉 平均 MAE：{np.mean(self.mae_list):.4f}")

            avg_precision = np.mean(self.precision_list, axis=0)
            avg_recall = np.mean(self.recall_list, axis=0)
            beta = 0.3
            beta2 = beta * beta  # β² = 0.09，标准 F-measure 定义
            fmeasure = (1 + beta2) * avg_precision * avg_recall / (beta2 * avg_precision + avg_recall + 1e-8)
            max_fmeasure = np.max(fmeasure)

            print(f"📈 最大 F-measure：{max_fmeasure:.4f}")
            print("ℹ️  注：F-measure为所有阈值下的最大值（论文标准计算方法，β²=0.3）")
            print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='ICON-P')
    parser.add_argument("--task", default='SOD')
    parser.add_argument("--ckpt", default='E:/PyCharm/project/ATMF_FRNet/checkpoint/ICON/ICON-P/ICON-P_edge_41')

    args = parser.parse_args()
    task = args.task
    model = args.model
    ckpt = args.ckpt

    print(args.model, args.ckpt)
    if args.task == "SOD":
        # 修改为绝对路径
        test_path = 'E:/PyCharm/project/ATMF_FRNet/datasets/DUTS/Test'
        t = Test(dataset, test_path, model, ckpt, task)
        t.save()
    elif args.task == "SOC":
        for path in ['datasets/SOC/SOC-AC', 'datasets/SOC/SOC-BO', 'datasets/SOC/SOC-CL', 'datasets/SOC/SOC-HO',
                     'datasets/SOC/SOC-MB', 'datasets/SOC/SOC-OC', 'datasets/SOC/SOC-OV', 'datasets/SOC/SOC-SC',
                     'datasets/SOC/SOC-SO']:
            t = Test(dataset, path, model, ckpt, task)
            t.save()
    elif args.task == "COD":
        for path in ['datasets/CHAMELEON/Test', 'datasets/CAMO/Test', 'datasets/COD10K/Test', 'datasets/CPD1K/Test']:
            t = Test(dataset, path, model, ckpt, task)
            t.save()
    else:
        inf_time = 0
        for path in ['datasets/SOD/Test/']:
            start = time.time()
            t = Test(dataset, path, model, ckpt, task)
            t.save()
            end = time.time()
            inf_time += (end - start)
        inf_per_image = inf_time / 300
        fps = 1 / inf_per_image
        print(f"inf_per_image: {inf_per_image:.4f}, fps: {fps:.2f}")2