#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import caffe
import cv2
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import logging
import argparse

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import _get_image_blob as get_image_blob
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
from utils.logger import config_logger

config_logger()

fg_classes = ('__bg__', 'parent', 'student')
pose_classes = ('listen', 'write', 'handup', 'positive', 'negative')

def draw_boxes(ax, box_probs, pred_boxes, pose_probs, head_pred_boxes, \
        color='r', linewidth=2):
    for i in range(len(box_probs)):
        sp_p = box_probs[i]
        b = pred_boxes[i]
        pose_p = pose_probs[i]
        head_b = head_pred_boxes[i]

        i_p = sp_p.argmax()
        p = sp_p[i_p]
        i_pp = pose_p.argmax()
        pp = pose_p[i_pp]

        c = npr.random(3)
        coords = (b[0], b[1]), b[2]-b[0]+1, b[3]-b[1]+1
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=c,
            linewidth=linewidth))
        # t = '{:s}_{:.2f} {:s}_{:.2f}'.format(
        #         fg_classes[i_p][:3], p, pose_classes[i_pp][:3], pp)
        t = '{:s}-{:s}'.format(
                fg_classes[i_p][:3], pose_classes[i_pp][:3])
        ax.text(b[0]-2, b[1]-2, t, size='small', color='r',
                bbox={'facecolor': c, 'alpha': 0.5, 'pad': 0.15})
        head_coords = (head_b[0], head_b[1]), head_b[2]-head_b[0]+1, head_b[3]-head_b[1]+1
        ax.add_patch(plt.Rectangle(*head_coords, fill=False, edgecolor=c,
            linewidth=1))

def detect(net, im):
    im_blob, im_scales = get_image_blob(im)
    assert len(im_scales )== 1, 'Only support single image test'
    blobs = {'data': im_blob}
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

    # forward
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False), 
            im_info=blobs['im_info'].astype(np.float32, copy=False))
    # fg
    box_probs = blobs_out['cls_prob']
    # fg box
    rois = net.blobs['rois'].data.copy()
    boxes = rois[:, 1:5] / im_scales[0]
    box_deltas = blobs_out['bbox_pred']
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.shape)
    # pose
    pose_probs = blobs_out['pose_cls_prob']
    # head bbox
    head_box_deltas = blobs_out['head_bbox_pred'][:, 4:]
    head_bbox_means = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
    head_bbox_stds = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS)
    head_box_deltas = head_box_deltas * head_bbox_stds + head_bbox_means
    head_pred_boxes = bbox_transform_inv(boxes, head_box_deltas)
    head_pred_boxes = clip_boxes(head_pred_boxes, im.shape)

    # NMS
    conf_thresh = 0.7
    nms_thresh = 0.5
    _max_inds = np.argmax(box_probs, axis=1)
    cond1 = _max_inds != 0
    cond2 = box_probs[range(len(_max_inds)), _max_inds] > conf_thresh
    fg_inds = np.where(np.logical_and(cond1, cond2))[0]
    cls_inds = _max_inds[fg_inds]
    fg_probs = box_probs[fg_inds, cls_inds]
    fg_boxes = pred_boxes[fg_inds].reshape((len(fg_inds), -1, 4))
    fg_boxes = fg_boxes[range(len(cls_inds)), cls_inds]
    dets = np.hstack((fg_boxes, fg_probs[:, None])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    logging.info('NMS: {} -> {}'.format(len(fg_inds), len(keep)))
    keep_inds = fg_inds[keep]
    fg_boxes = fg_boxes[keep]

    box_probs = box_probs[keep_inds]
    pose_probs = pose_probs[keep_inds]
    head_pred_boxes = head_pred_boxes[keep_inds]

    # show
    plt.imshow(im[:, :, [2, 1, 0]])
    draw_boxes(plt.gca(), box_probs, fg_boxes, pose_probs, head_pred_boxes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototxt', required=True)
    parser.add_argument('--caffemodel', required=True)
    parser.add_argument('image_path')
    args = parser.parse_args()

    cfg.TEST.HAS_RPN = True
    cfg.TEST.SCALES = (600,)
    cfg.TEST.RPN_MIN_SIZE = 10

    prototxt_p = args.prototxt
    weights_p = args.caffemodel
    image_p = args.image_path

    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt_p, weights_p, caffe.TEST)

    im = cv2.imread(image_p)
    plt.figure(figsize=(12, 8))
    detect(net, im)
    plt.axis('off')
    plt.tight_layout(pad=0.2)
    plt.savefig('det.png')
    print 'save to det.png'
