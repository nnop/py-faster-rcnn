#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import _init_paths
from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.train import get_training_roidb, train_net
from datasets.ft_body import FtBody
# from datasets.pascal_voc import pascal_voc


def _init_caffe(cfg):
    import caffe
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

if __name__ == "__main__":
    # config
    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TRAIN.BBOX_REG = False
    cfg.TRAIN.IMS_PER_BATCH = 1

    # get_imdb()
    imdb = FtBody('train')
    # imdb = pascal_voc('train', '2007')
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    roidb = get_training_roidb(imdb)
    output_dir = get_output_dir(imdb)
    print output_dir

    solver = 'models/ft_body/ZF/faster_rcnn_alt_opt/stage1_rpn_solver60k80k.pt'
    max_iters = 80000
    rpn_test_prototxt = 'models/ft_body/ZF/faster_rcnn_alt_opt/rpn_test.pt'
    init_model = 'data/imagenet_models/ZF.v2.caffemodel'

    _init_caffe(cfg)
    model_paths = train_net(solver, roidb, output_dir,
            pretrained_model=init_model, max_iters=max_iters)
    print model_paths
