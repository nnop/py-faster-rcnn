#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

# LOG="experiments/logs/ft_body_vgg16_end2end_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
LOG="union_end2end_train.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu 1 \
  --solver models/ft_union/VGG16/faster_rcnn_end2end/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb ft_union_train \
  --iters 70000 \
  --cfg experiments/cfgs/union_end2end.yml

# set +x
# NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
# set -x

# time ./tools/test_net.py --gpu 7 \
#   --def models/ft_union/VGG16/faster_rcnn_end2end/test.prototxt \
#   --net ${NET_FINAL} \
#   --imdb ft_union_test \
#   --cfg experiments/cfgs/union_end2end.yml \
