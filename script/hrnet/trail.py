#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 02 - 10
# @Author  : Luo jin
# @User    : jim 
# @File    : trail.py
# -----------------------------------------
#
import torch
from config import _C as config
from config import update_config
from seg.models.hrnet.hrnet_ocr import get_seg_model

if __name__ == '__main__':
    update_config(config,"./config.yaml")
    print(config)
    model=get_seg_model(config)
    model_ckpt = "F:/Download/edge download/hrnet_cs_8090_torch11.pth"
    model.load_state_dict(torch.load(model_ckpt))
    # images=torch.ones((1,3,256,256)).cuda()
    # output=model(images)
    # for item in output:
    #     print(item.shape)
