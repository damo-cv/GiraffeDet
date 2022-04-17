import numpy as np

cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.npy_float32, ndim=2] _match_coco(
        np.ndarray[np.npy_float32, ndim=2] iou_mat,
        np.ndarray[np.npy_float32, ndim=1] iou_thrs,
        np.ndarray[np.npy_bool, ndim=1] is_ignore,
        np.ndarray[np.npy_bool, ndim=1] is_crowd
):
    cdef int num_det, num_gt, num_iou_thr
    num_det, num_gt, num_iou_thr = iou_mat.shape[0], iou_mat.shape[1], \
                                   iou_thrs.shape[0]

    cdef np.ndarray[np.npy_int32, ndim=2] matched_gt = np.empty((num_iou_thr,
                                                                 num_det),
                                                                dtype=np.int32)
    cdef np.ndarray[np.npy_bool, ndim=2] gt_matched = np.zeros((num_iou_thr,
                                                                num_gt),
                                                               dtype=np.bool)
    cdef np.npy_float32 iou
    cdef np.npy_int32 matching_gt

    for iou_thr_idx in range(num_iou_thr):
        for det_idx in range(num_det):
            iou = iou_thrs[iou_thr_idx]
            iou_ignore = iou_thrs[iou_thr_idx]
            matching_gt = -1
            for gt_idx in range(num_gt):
                if gt_matched[iou_thr_idx, gt_idx] and (not is_crowd[gt_idx]):
                    continue
                # if dt matched to reg gt, and on ignore gt, stop
                if matching_gt > -1 and (not is_ignore[matching_gt]) and \
                        is_ignore[gt_idx]:
                    continue
                # continue to next gt unless better match made
                _iou = iou_ignore if is_ignore[gt_idx] else iou
                _iou_mat_val = iou_mat[det_idx, gt_idx]
                if _iou_mat_val < _iou:
                    continue
                # match successful
                _iou = iou_mat[det_idx, gt_idx]
                if is_ignore[gt_idx]:
                    iou_ignore = _iou_mat_val
                else:
                    iou = _iou_mat_val
                matching_gt = gt_idx
            if matching_gt != -1:
                gt_matched[iou_thr_idx, matching_gt] = True
            matched_gt[iou_thr_idx, det_idx] = matching_gt
    return matched_gt

def match_coco(iou_mat, iou_thrs, is_ignore, is_crowd):
    return _match_coco(iou_mat, iou_thrs, is_ignore, is_crowd)
