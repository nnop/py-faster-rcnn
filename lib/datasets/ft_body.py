import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
import scipy

from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import clip_boxes
from datasets.imdb import imdb
from pascal_voc import pascal_voc 

class ft_body(pascal_voc):
    def __init__(self, image_set, data_path=None):
        imdb.__init__(self, 'ft_body_'+image_set)
        self._image_set = image_set
        self._data_path = data_path if data_path else self._get_default_path()
        self._classes = ('__background__', 'parent', 'student')
        self._image_ext = '.jpg'
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        self._image_index = self._load_image_set_index()
        assert osp.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def gt_roidb(self):
        """
        Return the database of ground-truth ROIs.

        TODO: add cache file
        """

        gt_roidb = [self._load_ftbody_annotation(idx)
                for idx in self.image_index]
        return gt_roidb

    def _get_default_path(self):
        """
        Return the default dataset root path.
        """
        return osp.join(cfg.DATA_DIR, 'ft_body')

    def _load_ftbody_annotation(self, idx):
        """
        Load image and bboxes from XML file.
        """
        anno_p = osp.join(self._data_path, 'Annotations', idx+'.xml')
        tree = ET.parse(anno_p)
        objs = tree.findall('object')
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs,), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # image meta info
        size_obj = tree.find('size')
        im_wid = int(size_obj.find('width').text)
        im_hei = int(size_obj.find('height').text)
        im_dep = int(size_obj.find('depth').text)
        filename = tree.find('filename').text

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            # assert 0 <= x1 < x2 <= im_wid and 0 <= y1 < y2 <= im_hei, \
            #         '{} (h:{}, w:{}) {}'.format(idx+'.xml', im_hei, im_wid, [x1, y1, x2, y2])
            cls_name = obj.find('name').text.lower().strip()
            cls = self._class_to_ind[cls_name]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
        boxes = clip_boxes(boxes, (im_hei, im_wid))
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'width': im_wid,
                'height': im_hei,
                'filename': filename,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}

