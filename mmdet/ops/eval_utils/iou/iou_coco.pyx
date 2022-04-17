import numpy as np

cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.npy_float32, ndim=2] _iou_coco(
        np.ndarray[np.npy_float32, ndim=2] det_boxes,
        np.ndarray[np.npy_float32, ndim=2] gt_boxes,
        np.ndarray[np.npy_bool, ndim=1] is_crowd
):
    cdef int num_det, num_gt, num_iou_thr
    num_det, num_gt = det_boxes.shape[0], gt_boxes.shape[0]

    cdef np.ndarray[np.npy_float32, ndim=2] det_boxes_tl, det_boxes_br
    cdef np.ndarray[np.npy_float32, ndim=2] gt_boxes_tl, gt_boxes_br

    cdef np.ndarray[np.npy_float32, ndim=1] det_boxes_area, gt_boxes_area
    cdef np.ndarray[np.npy_float32, ndim=3] inter_tl, inter_br

    det_boxes_tl, det_boxes_br = det_boxes[:, :2], det_boxes[:, 2:]
    gt_boxes_tl, gt_boxes_br = gt_boxes[:, :2], gt_boxes[:, 2:]

    det_boxes_area = (det_boxes_br - det_boxes_tl).prod(axis=-1)
    gt_boxes_area = (gt_boxes_br - gt_boxes_tl).prod(axis=-1)

    inter_tl = np.maximum(det_boxes_tl[:, None, :], gt_boxes_tl[None, :, :])
    inter_br = np.minimum(det_boxes_br[:, None, :], gt_boxes_br[None, :, :])

    cdef np.ndarray[np.npy_float32, ndim=2] iou_mat
    iou_mat = np.zeros((num_det, num_gt), dtype=np.float32)
    cdef float tlx, tly, brx, bry, inter_area, union_area
    for det_idx in range(num_det):
        for gt_idx in range(num_gt):
            tlx = inter_tl[det_idx, gt_idx, 0]
            tly = inter_tl[det_idx, gt_idx, 1]
            brx = inter_br[det_idx, gt_idx, 0]
            bry = inter_br[det_idx, gt_idx, 1]
            if tlx >= brx or tly >= bry:
                iou_mat[det_idx, gt_idx] = 0.
            else:
                inter_area = (brx - tlx) * (bry - tly)
                if is_crowd[gt_idx]:
                    union_area = det_boxes_area[det_idx]
                else:
                    union_area = det_boxes_area[det_idx] + gt_boxes_area[
                        gt_idx] - inter_area
                if union_area <= 0.:
                    union_area = 1e-7
                iou_mat[det_idx, gt_idx] = inter_area / union_area

    return iou_mat

def iou_coco(det_boxes, gt_boxes, is_crowd):
    return _iou_coco(det_boxes, gt_boxes, is_crowd)
