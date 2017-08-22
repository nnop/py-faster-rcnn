#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import matplotlib
matplotlib.use('Agg')
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import scipy.io as sio
import caffe, os, sys, cv2
import os.path as osp
import argparse

CLASSES = ('__background__',
           'parent', 'student')

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    print 'found {} {}'.format(len(inds), class_name)
    if len(inds) == 0:
        return

    ax = plt.gca()
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=npr.random(3), linewidth=1.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.1f}'.format(class_name[:1], score),
                bbox=dict(facecolor='blue', alpha=0.5),
                color='white')

def demo(net, im_file):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)
    im = im[:, :, (2, 1, 0)]
    fig = plt.figure(figsize=(18, 12))
    plt.imshow(im, aspect='equal')

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.4
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    plt.axis('off')
    plt.tight_layout()

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.RPN_MIN_SIZE = 10
    cfg.TEST.RPN_NMS_THRESH = 0.5

    prototxt = 'models/ft_body/VGG16/faster_rcnn_end2end/test.prototxt'
    caffemodel = 'output/body_end2end/ft_body_train/vgg16_rpn_body_iter_70000.caffemodel'
    im_file = 'data/ft_body_whole/JPEGImages/0612-41-1000.jpg'

    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    demo(net, im_file)

    out_fn = osp.split(im_file)[1][:-4] + '_det.png'
    plt.savefig(out_fn)
    print 'save to', out_fn

    # plt.show()