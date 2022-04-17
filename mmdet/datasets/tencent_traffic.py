import os.path as osp
import glob

import numpy as np
from .custom import CustomDataset

from .builder import DATASETS


@DATASETS.register_module()
class TrafficSignDataset(CustomDataset):
    CLASSES = ('sign',)

    def __init__(self, *args, **kwargs):
        self.difficulty_thresh = kwargs.pop('difficulty_thresh', 100)
        # set the default threshold to a big value. we take all gts as not difficult by default
        super(TrafficSignDataset, self).__init__(*args, **kwargs)

    def load_annotations(self, ann_folder):
        """
            Params:
                ann_folder: folder that contains tencent annotations txt files
        """
        ann_files = glob.glob(ann_folder + '/*.circle')
        data_infos = []
        if not ann_files:  # test phase, use image folder as ann_folder to generate pseudo annotations
            ann_files = glob.glob(ann_folder + '/*.jpg')
            for ann_file in ann_files:
                data_info = dict()
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = dict()
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = dict()
                img_id = osp.split(ann_file)[1][:-7]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = dict()
                gt_bboxes = []
                gt_labels = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.strip().split(',')
                        bbox = bbox_info[4:]
                        bbox = [*map(lambda x: float(x), bbox)]
                        x, y, w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                        bbox = [x, y, x + w, y + h]

                        difficulty = 0
                        label = 0

                        if difficulty >= self.difficulty_thresh:
                            gt_labels_ignore.append(label)
                            gt_bboxes_ignore.append(bbox)
                        else:
                            gt_labels.append(label)
                            gt_bboxes.append(bbox)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)

                else:
                    data_info['ann']['bboxes'] = np.zeros(
                        (0, 4), dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)

                if gt_bboxes_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)

                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 4), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if data_info['ann']['labels'].size > 0:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
