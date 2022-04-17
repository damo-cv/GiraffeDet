import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class YOLOV4BBoxCoder(BaseBBoxCoder):
    """YOLO BBox coder.

    Following `YOLO <https://arxiv.org/abs/1506.02640>`_, this coder divide
    image into grids, and encode bbox (x1, y1, x2, y2) into (cx, cy, dw, dh).
    cx, cy in [0., 1.], denotes relative center position w.r.t the center of
    bboxes. dw, dh are the same as :obj:`DeltaXYWHBBoxCoder`.

    Args:
        eps (float): Min value of cx, cy when encoding.
    """

    def __init__(self, eps=1e-6):
        super(BaseBBoxCoder, self).__init__()
        self.eps = eps

    def encode(self, bboxes, gt_bboxes, stride):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., anchors.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.
            stride (torch.Tensor | int): Stride of bboxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        raise NotImplementedError

    def decode(self, bboxes, pred_bboxes, stride):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes, e.g. anchors.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            stride (torch.Tensor | int): Strides of bboxes.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        # Get outputs x, y
        x_center_pred = pred_bboxes[..., 0] * stride + x_center
        y_center_pred = pred_bboxes[..., 1] * stride + y_center
        w_pred = pred_bboxes[..., 2] * w
        h_pred = pred_bboxes[..., 3] * h

        decoded_bboxes = torch.stack(
            (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
             x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
            dim=-1)

        return decoded_bboxes
