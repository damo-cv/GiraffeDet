# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.runner import force_fp32
from mmcv.runner.fp16_utils import auto_fp16

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply,
                        multiclass_nms)
from ..builder import HEADS, LOSSES, build_loss
from ..losses import reduce_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@LOSSES.register_module()
class SoftFocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn()
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, raw_loss, gamma=1.5, alpha=0.25):
        super(SoftFocalLoss, self).__init__()
        self.loss_fcn = build_loss(raw_loss)  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = raw_loss.reduction
        # required to apply FL to each element
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, gt, reduction_override=None):
        loss = self.loss_fcn(pred, gt)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma
        # non-zero power for gradient stability

        # TF implementation
        # https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = gt * pred_prob + (1 - gt) * (1 - pred_prob)
        alpha_factor = gt * self.alpha + (1 - gt) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t)**self.gamma
        loss *= alpha_factor * modulating_factor
        if reduction_override is not None:
            return reduce_loss(loss, reduction=reduction_override)
        else:
            return reduce_loss(loss, reduction=self.reduction)


@HEADS.register_module()
class YOLOCSPHead(BaseDenseHead, BBoxTestMixin):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): Number of input channels per scale.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        featmap_strides (List[int]): The stride of each scale.
            Should be in descending order. Default: (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Default: 0.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        loss_cls (dict): Config of classification loss.
        loss_conf (dict): Config of confidence loss.
        loss_xy (dict): Config of xy coordinate loss.
        loss_wh (dict): Config of wh coordinate loss.
        train_cfg (dict): Training config of YOLOV3 head. Default: None.
        test_cfg (dict): Testing config of YOLOV3 head. Default: None.
    """

    def __init__(
            self,
            num_classes,
            in_channels,
            anchor_generator=dict(
                type='YOLOV4AnchorGenerator',
                base_sizes=[
                    [(12, 16), (19, 36), (40, 28)],  # P3/8
                    [(36, 75), (76, 55), (72, 146)],  # P4/16
                    [(142, 110), (192, 243), (459, 401)]
                ],  # P5/32
                strides=[8, 16, 32]),
            bbox_coder=dict(type='YOLOV4BBoxCoder'),
            featmap_strides=[8, 16, 32],
            one_hot_smoother=0.,
            conv_cfg=None,
            norm_cfg=dict(
                type='BN', requires_grad=True, eps=0.001, momentum=0.03),
            act_cfg=dict(type='Mish'),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=32.),
            loss_conf=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=64.),
            loss_bbox=dict(type='GIoULoss', loss_weight=3.2),
            class_agnostic=False,
            train_cfg=None,
            test_cfg=None,
            init_cfg=None):

        super(YOLOCSPHead, self).__init__()
        # Check params
        assert (len(in_channels) == len(featmap_strides))

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.featmap_strides = featmap_strides
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner = None
        self.sampler = None
        self.shape_match_thres = 4.
        self.conf_iou_loss_ratio = 1.
        self.conf_level_balance_weight = [4.0, 1.0, 0.4, 0.1, 0.1]
        self.class_freq = None
        self.num_obj_avg = 8
        if self.train_cfg is not None:
            if hasattr(self.train_cfg, 'assigner'):
                self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
            if hasattr(self.train_cfg, 'conf_iou_loss_ratio'):
                self.conf_iou_loss_ratio = self.train_cfg.conf_iou_loss_ratio
            if hasattr(self.train_cfg, 'conf_level_balance_weight'):
                self.conf_level_balance_weight = \
                    self.train_cfg.conf_level_balance_weight
            if hasattr(self.train_cfg, 'class_frequency'):
                self.class_freq = self.train_cfg.class_frequency
            if hasattr(self.train_cfg, 'num_obj_per_image'):
                self.num_obj_avg = self.train_cfg.num_obj_per_image
            if hasattr(self.train_cfg, 'shape_match_thres'):
                self.shape_match_thres = self.train_cfg.shape_match_thres

        self.one_hot_smoother = one_hot_smoother

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.anchor_generator = build_anchor_generator(anchor_generator)

        self.class_agnostic = class_agnostic

        if not self.class_agnostic:
            self.loss_cls = build_loss(loss_cls)
        self.loss_conf = build_loss(loss_conf)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_bbox_weight = self.loss_bbox.loss_weight
        self.loss_bbox.loss_weight = 1.

        self.num_anchors = self.anchor_generator.num_base_anchors
        self._init_layers()
        self.fp16_enabled = False

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        if not self.class_agnostic:
            return 5 + self.num_classes
        else:
            return 5

    def _init_layers(self):
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Conv2d(self.in_channels[i],
                                  self.num_anchors[i] * self.num_attrib, 1)
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.convs_pred:
            normal_init(m, std=0.01)
        for m, stride in zip(self.convs_pred, self.featmap_strides):
            b = m.bias.view(-1, self.num_attrib)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(
                self.num_obj_avg /
                (640 / stride)**2)  # obj (8 objects per 640 image)
            if not self.class_agnostic:
                b[:, 5:] += math.log(
                    0.6 / (self.num_classes -
                           0.99)) if self.class_freq is None else torch.log(
                               self.class_freq / self.class_freq.sum())  # cls
            m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    @auto_fp16()
    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        assert len(feats) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            pred_map = self.convs_pred[i](feats[i])
            pred_maps.append(pred_map)

        return tuple(pred_maps),

    # @force_fp32(apply_to=('pred_maps',))
    def get_bboxes(self,
                   pred_maps,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        num_levels = len(pred_maps)
        num_image = len(img_metas)

        featmap_sizes = [pred_maps[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, pred_maps[0].device)

        mlvl_bbox_pred = []
        mlvl_conf_pred = []
        mlvl_score_pred = []

        for lvl in range(num_levels):
            lvl_pred_maps = pred_maps[lvl].permute(0, 2, 3, 1).reshape(
                (num_image, -1, self.num_attrib))
            # activation
            lvl_pred_maps = lvl_pred_maps.sigmoid()
            # class score
            if not self.class_agnostic:
                mlvl_score_pred.append(lvl_pred_maps[:, :, 5:])
            # conf score
            mlvl_conf_pred.append(lvl_pred_maps[:, :, 4])
            # bbox transform
            lvl_pred_maps[:, :, :2] = lvl_pred_maps[:, :, :2] * 2. - 1.
            lvl_pred_maps[:, :, 2:4] = (lvl_pred_maps[:, :, 2:4] * 2)**2
            lvl_bbox_pred = lvl_pred_maps[:, :, :4].reshape(-1, 4)
            lvl_anchors = mlvl_anchors[lvl][None, ...].repeat(
                (num_image, 1, 1))

            lvl_bbox_pred = self.bbox_coder.decode(
                bboxes=lvl_anchors.reshape(-1, 4),
                pred_bboxes=lvl_bbox_pred.reshape(-1, 4),
                stride=self.featmap_strides[lvl])
            lvl_bbox_pred = lvl_bbox_pred.reshape((num_image, -1, 4))
            mlvl_bbox_pred.append(lvl_bbox_pred)

        if not self.class_agnostic:
            mimg_score_pred = [
                score for score in torch.cat(mlvl_score_pred, dim=1)
            ]
        else:
            mimg_score_pred = None
        mimg_conf_pred = [conf for conf in torch.cat(mlvl_conf_pred, dim=1)]
        mimg_bbox_pred = [bbox for bbox in torch.cat(mlvl_bbox_pred, dim=1)]

        result_list = []

        for img_id in range(len(img_metas)):
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(
                cls_pred=mimg_score_pred[img_id]
                if mimg_score_pred is not None else None,
                conf_pred=mimg_conf_pred[img_id],
                bbox_pred=mimg_bbox_pred[img_id],
                scale_factor=scale_factor,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_pred,
                           conf_pred,
                           bbox_pred,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_pred (Tensor): Score of each predicted bbox of a single image.
            conf_pred (Tensor): Confidence of each predicted bbox of a single
                image.
            bbox_pred (Tensor): Predicted bbox of a single image.
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg

        # Get top-k prediction
        nms_pre = cfg.get('nms_pre', -1)

        if 0 < nms_pre < conf_pred.size(0):
            _, topk_inds = conf_pred.topk(nms_pre)
            bbox_pred = bbox_pred[topk_inds, :]
            if not self.class_agnostic:
                cls_pred = cls_pred[topk_inds, :]
            conf_pred = conf_pred[topk_inds]

        if not self.class_agnostic:
            cls_pred *= conf_pred[:, None]
        else:
            cls_pred = conf_pred[:, None]

        if with_nms and (cls_pred.size(0) == 0):
            return torch.zeros((0, 5)), torch.zeros((0, ))

        if rescale:
            bbox_pred /= bbox_pred.new_tensor(scale_factor)

        if with_nms:
            # In mmdet 2.x, the class_id for background is num_classes.
            # i.e., the last column.
            padding = cls_pred.new_zeros(cls_pred.shape[0], 1)
            cls_pred = torch.cat([cls_pred, padding], dim=1)

            det_bboxes, det_labels = multiclass_nms(bbox_pred, cls_pred,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            cls_pred = cls_pred * conf_pred[:, None]
            class_score, class_id = cls_pred.max(dim=-1)
            return torch.cat((bbox_pred, class_score[:, None]),
                             dim=-1), class_id

    @force_fp32(apply_to=('pred_maps', ))
    def loss(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_gts = pred_maps[0].new_tensor([g.size(0)
                                           for g in gt_bboxes]).mean()

        pred_maps = [p.float() for p in pred_maps]

        device = pred_maps[0][0].device

        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]

        responsible_indices = self.anchor_generator.responsible_indices(
            featmap_sizes,
            gt_bboxes,
            neighbor=2 if self.assigner is None else 3,
            shape_match_thres=self.shape_match_thres,
            device=device)

        if self.assigner is None:
            results = self.get_targets_no_assigner(responsible_indices,
                                                   gt_bboxes, gt_labels)

            (mlvl_pos_indices, mlvl_gt_bboxes_targets,
             mlvl_gt_labels_targets) = results

            mlvl_anchors = self.anchor_generator.grid_anchors(
                featmap_sizes, device)

            # valid_flag_list = []
            # for img_id, img_meta in enumerate(img_metas):
            #     multi_level_flags = self.anchor_generator.valid_flags(
            #         featmap_sizes, img_meta['pad_shape'], device)
            #     valid_flag_list.append(multi_level_flags)

            losses_cls, losses_conf, losses_bbox = multi_apply(
                self.loss_single_no_assigner, pred_maps, mlvl_anchors,
                self.featmap_strides, mlvl_pos_indices, mlvl_gt_bboxes_targets,
                mlvl_gt_labels_targets)
        else:
            raise NotImplementedError

        losses_conf = [
            loss_conf * balance for loss_conf, balance in zip(
                losses_conf, self.conf_level_balance_weight)
        ]

        if not self.class_agnostic:
            return dict(
                loss_cls=losses_cls,
                loss_conf=losses_conf,
                loss_bbox=losses_bbox,
                num_gts=num_gts)
        else:
            return dict(
                loss_conf=losses_conf, loss_bbox=losses_bbox, num_gts=num_gts)

    def loss_single_no_assigner(self, pred_map, anchors, stride, pos_indices,
                                target_bboxes, target_labels):
        """Compute loss of a single level.

        Args:
            pred_map (Tensor): Raw predictions for a single level.
            anchors (Tensor): anchors for a single level.
                has shape (num_anchors, 4)
            pos_indices (tuple[Tensor]): positive sample indices.
                (img_idx, anchor_idx), each indices tensor has shape
                (k,), which stands for k positive samples
            pos_bboxes (Tensor): target tensor for positive samples.
                has shape (k, 4)
            pos_labels (Tensor): target tensor for positive samples.
                has shape (k, self.num_classes)

        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """

        num_imgs = len(pred_map)
        pred_map = pred_map.permute(0, 2, 3,
                                    1).reshape(num_imgs, -1, self.num_attrib)

        img_ind, anchor_ind = pos_indices

        pred_conf = pred_map[..., 4]
        target_conf = torch.zeros_like(pred_conf, requires_grad=False)

        loss_bbox = pred_map.new_zeros((1, ))
        loss_cls = pred_map.new_zeros((1, ))

        if anchor_ind.numel():
            pred_map_pos = pred_map[img_ind, anchor_ind]
            anchor_pos = anchors[anchor_ind]

            # apply transforms on bbox prediction
            pred_bbox = pred_map_pos[..., :4].sigmoid()
            pred_bbox_xy = pred_bbox[..., :2] * 2. - 1.
            pred_bbox_wh = (pred_bbox[..., 2:] * 2.)**2.
            pred_bbox = self.bbox_coder.decode(
                anchor_pos, torch.cat((pred_bbox_xy, pred_bbox_wh), dim=-1),
                stride)

            giou_loss = self.loss_bbox(
                pred_bbox, target_bboxes, reduction_override='none')

            loss_bbox += reduce_loss(
                giou_loss, reduction=self.loss_bbox.reduction)

            pred_cls = pred_map_pos[..., 5:]
            target_cls = target_labels

            if not self.class_agnostic:
                loss_cls += self.loss_cls(pred_cls, target_cls)

            _r = self.conf_iou_loss_ratio
            _conf_t = (1 - _r) + _r * (1 - giou_loss).detach().clamp(0.0, 1.0)
            target_conf[pos_indices] = _conf_t.type(target_conf.dtype)

        loss_conf = self.loss_conf(pred_conf, target_conf)

        return loss_cls, loss_conf, loss_bbox * self.loss_bbox_weight

    def get_targets_no_assigner(self, responsible_indices_list, gt_bboxes_list,
                                gt_labels_list):
        """Compute target maps for anchors in multiple images.

        Args:
            responsible_indices_list ([list[tuple[Tensor]]]): Multi level
                responsible indices. Each element is a tuple of 3 tensors of
                shape (m,), (img_id, anchor_id, img_gt_id)
                where m stands for the number of matches
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.

        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - mlvl_pos_indices (tuple[Tensor, Tensor]): Postive
                    image/anchor indices of each level.
                - mlvl_gt_bboxes_targets (list[Tensor]): Gt bbox target
                    corresponds to the aforementioned indices.
                - mlvl_gt_labels_targets (list[Tensor]): Gt label corresponds
                    to the aforementioned indices.
        """

        mlvl_pos_indices = [None for _ in range(self.num_levels)]
        mlvl_gt_bboxes_targets = [None for _ in range(self.num_levels)]
        mlvl_gt_labels_targets = [None for _ in range(self.num_levels)]

        gt_bboxes = torch.cat(gt_bboxes_list, dim=0)
        gt_labels = torch.cat(gt_labels_list, dim=0)

        for lvl in range(self.num_levels):
            img_ind, anchor_ind, gt_ind = responsible_indices_list[lvl]

            mlvl_pos_indices[lvl] = (img_ind, anchor_ind)
            mlvl_gt_bboxes_targets[lvl] = gt_bboxes[gt_ind]
            gt_labels_targets = gt_labels[gt_ind]

            gt_labels_targets = F.one_hot(
                gt_labels_targets, num_classes=self.num_classes).float()
            if self.one_hot_smoother != 0:  # label smooth
                gt_labels_targets = gt_labels_targets * (
                    1 - self.one_hot_smoother
                ) + self.one_hot_smoother / self.num_classes
            mlvl_gt_labels_targets[lvl] = gt_labels_targets

        return mlvl_pos_indices, mlvl_gt_bboxes_targets, mlvl_gt_labels_targets

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
