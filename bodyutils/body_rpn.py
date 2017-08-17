#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import caffe
import numpy as np
import numpy.random as npr
import os.path as osp

import sys
sys.path.insert(0, 'lib')

from fast_rcnn.config import cfg
from fast_rcnn.test import _get_blobs

def filter_box_size(boxes):
    heights = boxes[:, 3] - boxes[:, 1]
    boxes = boxes[heights < 400]
    return boxes

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=npr.random(3), linewidth=2)
            )
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    # plt.axis('off')
    plt.tight_layout()
    plt.draw()

if __name__ == "__main__":
    # im_p = '10.jpg'
    im_p = '/home/disk2/yeqiu/bag_result/0722_409/raw/111660.jpg'
    model_p = 'vgg16_rpn_test.pt'
    weights_p = 'output/ft_body2/ft_body_train/vgg16_rpn_stage1_iter_80000.caffemodel'

    net = caffe.Net(model_p, weights_p, caffe.TEST)

    cfg.TEST.SCALES = (600,)
    cfg.TEST.HAS_RPN = True
    cfg.TEST.RPN_NMS_THRESH = 0.3
    cfg.TEST.RPN_PRE_NMS_TOP_N = 800
    cfg.TEST.RPN_POST_NMS_TOP_N = 40

    # get blobs
    im = cv2.imread(im_p)
    blobs, im_scales = _get_blobs(im, None)

    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)

    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    rois = net.blobs['rois'].data.copy()
    scores = net.blobs['scores'].data.copy()

    # unscale back to raw image space
    boxes = rois[:, 1:5] / im_scales[0]
    boxes = filter_box_size(boxes)
    vis_detections(im, 'person', boxes, 0.9)

    # save
    main_name = osp.splitext(osp.split(im_p)[1])[0]
    plt.savefig(main_name+'_res.png')

