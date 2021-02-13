#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 02 - 09
# @Author  : Luo jin
# @User    : 22403 
# @File    : train.py
# -----------------------------------------
#
import os
import torch
import datetime
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from seg.data.dataloader import Loader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from seg.models.networks.nets.unet import UNet
from seg.loss.dice import GeneralizedDiceLoss, GeneralizedWassersteinDiceLoss, LogCoshGeneralizedDiceLoss
from seg.loss.focal_loss import FocalLoss
from seg.loss.tversky import TverskyLoss
from seg.metrics.mIOU import IOUMetric

if __name__ == '__main__':
    EPOCHS = 50
    InteLog = 10
    schedulerStep = 9
    batch_size = 28
    # ----------------------------------------------------------------------------------------------
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_path = os.path.join('../checkpoint/unet', current_time)
    if not os.path.exists(current_path): os.mkdir(current_path)
    model_dir = os.path.join(current_path, 'models')
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    log_dir = os.path.join(current_path, 'logs')
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)
    # ----------------------------------------------------------------------------------------------
    txtPath = '../train2.txt'
    datasets = Loader(txtPath)
    feeder = DataLoader(datasets, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(),
                        drop_last=True, num_workers=6)
    STEPS = len(feeder)
    # ----------------------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        dimensions=2,
        in_channels=4,
        out_channels=10,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    model_ckpt="../checkpoint/unet/20210212_215043/models/state_dict_model_e_0.pt"
    model.load_state_dict(torch.load(model_ckpt))
    model.train()
    model.cuda()
    # ----------------------------------------------------------------------------------------------
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    # ----------------------------------------------------------------------------------------------
    GDiceLoss = GeneralizedDiceLoss(softmax=True)
    LCGDiceLoss = LogCoshGeneralizedDiceLoss(softmax=True)
    Focal_loss = FocalLoss(gamma=8.0)
    Tversky_loss = TverskyLoss(softmax=True,alpha=0.7, beta=0.3)
    # ----------------------------------------------------------------------------------------------
    metirc = IOUMetric(num_classes=10)
    # ----------------------------------------------------------------------------------------------
    for epoch in range(EPOCHS):
        print("curent learning rate is ", optimizer.param_groups[0]["lr"])
        for step, (images, labels) in enumerate(feeder):
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            preds = model(images)
            # loss = GDiceLoss(preds, labels)
            Floss=Focal_loss(preds,labels)
            Tloss=Tversky_loss(preds,labels)
            loss=Floss+Tloss
            if step % InteLog == 0:
                pred_index = np.argmax(preds.cpu().detach().numpy(), axis=1)
                label_index = np.argmax(labels.cpu().detach().numpy(), axis=1)
                metirc.add_batch(pred_index, label_index)
                _, _, ius, mean_iu, _ = metirc.evaluate()
                loss_scalar = loss.cpu().detach().numpy()
                print('Epoch [{}][{}/{}]: loss: {:.6f}  mIOU: {:.6f}'.format(epoch + 1, step, STEPS, loss_scalar,
                                                                             mean_iu))
                print(
                    "ious: 0:{:.3f} 1:{:.3f} 2:{:.3f} 3:{:.3f} 4:{:.3f} 5:{:.3f} 6:{:.3f} 7:{:.3f} 8:{:.3f} 9:{:.3f} ".format(
                        ius[0], ius[1], ius[2], ius[3], ius[4], ius[5], ius[6], ius[7], ius[8], ius[9]))
                writer.add_scalar('loss', loss_scalar, epoch * STEPS + step)
                writer.add_scalar('mIOU', mean_iu, epoch * STEPS + step)
                for i in range(len(ius)):
                    section = 'IOU/%d' % i
                    writer.add_scalar(section, ius[i], epoch * STEPS + step)

            loss.backward()
            optimizer.step()
        if (epoch + 1) % schedulerStep == 0: scheduler.step()
        model_subdir = "state_dict_model_e_%d.pt" % epoch
        model_save_name = os.path.join(model_dir, model_subdir)
        torch.save(model.state_dict(), model_save_name)
    writer.close()
