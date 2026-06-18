import os
import sys
import datetime

sys.path.insert(0, '../')
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import dataset
import argparse
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter          # 或可改回 torch.utils.tensorboard

from model.get_model import get_model


# IoU Loss
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


# Structure Loss
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


# 边界提取函数（方案三）
def get_edge(mask):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    edge_x = F.conv2d(mask, sobel_x, padding=1)
    edge_y = F.conv2d(mask, sobel_y, padding=1)
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    edge = (edge > 0.1).float()
    return edge


def train(Dataset, parser):
    args = parser.parse_args()
    _MODEL_ = args.model
    _DATASET_ = args.dataset
    _LR_ = args.lr
    _DECAY_ = args.decay
    _MOMEN_ = args.momen
    _BATCHSIZE_ = args.batchsize
    _EPOCH_ = args.epoch
    _LOSS_ = args.loss
    _SAVEPATH_ = args.savepath
    _VALID_ = args.valid

    print(args)

    # ---- 绝对路径配置（避免工作目录问题）----
    PROJECT_ROOT = 'E:/PyCharm/project/ATMF_FRNet'
    # 训练集路径
    if not os.path.isabs(_DATASET_):
        _DATASET_ = os.path.join(PROJECT_ROOT, _DATASET_)
    # 模型保存路径
    if not os.path.isabs(_SAVEPATH_):
        _SAVEPATH_ = os.path.join(PROJECT_ROOT, _SAVEPATH_)

    if _MODEL_ in ["ICON-S", "ICON-P", "ICON-R", "ICON-V", "ICON-M"]:
        cfg = Dataset.Config(datapath=_DATASET_, savepath=_SAVEPATH_, mode='train', batch=_BATCHSIZE_, lr=_LR_,
                             momen=_MOMEN_, decay=_DECAY_, epoch=_EPOCH_)
    else:
        print("_MODEL_ IS NOT FOUND.")
        return

    data = Dataset.Data(cfg, _MODEL_)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True,
                        num_workers=0)

    net = get_model(cfg, _MODEL_)
    net.train(True)
    net.cuda()

    base, head = [], []
    for name, param in net.named_parameters():
        if 'encoder.conv1' in name or 'encoder.bn1' in name:
            pass
        elif 'encoder' in name:
            base.append(param)
        elif 'network' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)

    scaler = torch.cuda.amp.GradScaler()
    sw = SummaryWriter(_SAVEPATH_)   # TensorBoard 日志，若不想用可删除此行
    global_step = 0

    # 验证集绝对路径
    valid_paths = [
        os.path.join(PROJECT_ROOT, 'datasets/ECSSD/Test'),
        os.path.join(PROJECT_ROOT, 'datasets/PASCAL-S/Test')
    ]

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr

        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda(), mask.cuda()

            out1, out2, out3, out4, edge_pred = net(image)   # 5个输出

            with torch.cuda.amp.autocast():
                if _LOSS_ == "CPR":
                    loss1 = F.binary_cross_entropy_with_logits(out1, mask) + iou_loss(out1, mask)
                    loss2 = F.binary_cross_entropy_with_logits(out2, mask) + iou_loss(out2, mask)
                    loss3 = F.binary_cross_entropy_with_logits(out3, mask) + iou_loss(out3, mask)
                    loss4 = F.binary_cross_entropy_with_logits(out4, mask) + iou_loss(out4, mask)
                elif _LOSS_ == "STR":
                    loss1 = structure_loss(out1, mask)
                    loss2 = structure_loss(out2, mask)
                    loss3 = structure_loss(out3, mask)
                    loss4 = structure_loss(out4, mask)

                loss_sal = loss1 + loss2 + loss3 + loss4
                edge_gt = get_edge(mask)
                loss_edge = F.binary_cross_entropy_with_logits(edge_pred, edge_gt)
                loss = loss_sal + 0.1 * loss_edge

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {
                'loss1': loss1.item(), 'loss2': loss2.item(), 'loss3': loss3.item(), 'loss4': loss4.item(),
                'loss_edge': loss_edge.item()
            }, global_step=global_step)

            if step % 10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss1=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss_edge=%.6f'
                      % (datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],
                         loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss_edge.item()))

        if epoch > -1:
            # 保存模型，文件名加入 '_edge_' 标识
            save_name = os.path.join(_SAVEPATH_, _MODEL_ + '_edge_' + str(epoch + 1))
            torch.save(net.state_dict(), save_name)

            # 在每个验证集上运行预测（用于后续指标计算）
            for v_path in valid_paths:
                t = Valid(dataset, v_path, epoch, _MODEL_, _SAVEPATH_, "SOD")
                t.save()

            # Windows 下无法直接运行 .sh 脚本，故注释掉
            # cmd = os.path.join(os.getcwd().split('main')[0], "util/evaltool/run_sod_valid.sh")
            # os.system('{} {}'.format('sh', cmd + ' ICON-valid-' + str(epoch + 1)))


class Valid(object):
    def __init__(self, Dataset, Path, epoch, model_name, checkpoint_path, task=None):
        # checkpoint_path 应为绝对路径，且包含模型文件夹，例如 .../ICON-P/
        model_file = os.path.join(checkpoint_path, model_name + '_edge_' + str(epoch + 1))
        self.cfg = Dataset.Config(datapath=Path,
                                  snapshot=model_file,
                                  mode='test')
        self.data = Dataset.Data(self.cfg, model_name)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        self.net = get_model(self.cfg, model_name)
        self.net.train(False)
        self.net.cuda()
        self.epoch = epoch
        self.task = task

    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape = image.cuda().float(), (H, W)
                out1, out2, out3, out4, _ = self.net(image, shape)
                pred = torch.sigmoid(out4[0, 0]).cpu().numpy() * 255
                if self.task == "SOC":
                    head = 'util/evaltool/Prediction/ICON-valid-' + str(self.epoch + 1) + '/SOC/' + \
                           self.cfg.datapath.split('/')[-1]
                elif self.task == "SOD":
                    head = 'util/evaltool/Prediction/ICON-valid-' + str(self.epoch + 1) + '/' + \
                           self.cfg.datapath.split('/')[-2]
                else:
                    print("WRONG!")
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='ICON-P')
    parser.add_argument("--dataset", default='datasets/DUTS/Train')
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momen", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=1e-4)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--loss", default='CPR')
    parser.add_argument("--savepath", default='checkpoint/ICON/ICON-P/')
    parser.add_argument("--valid", default=True)
    train(dataset, parser)